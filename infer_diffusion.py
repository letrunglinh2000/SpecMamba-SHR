"""
Standalone inference for SpecMamba + Qformer/AnyIR.

Two paradigms (select with --paradigm):
  - pcc       : single forward pass at t=T (matches PCC training). DEFAULT.
  - diffusion : T-step DDIM-style sampling (legacy diffusion training).

Runs on arbitrary input images (no GT required). Supports single-GPU and
multi-GPU (via torchrun — files are sharded across ranks).

Usage (PCC, the current paradigm):
    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=29501 infer_unified.py \
        --input  /data2/letrunglinh/despecular/diff-train-style/SHDR-E2E-SHDNet/trainsets/SD2VAU/test/Inp \
        --gt     /data2/letrunglinh/despecular/diff-train-style/SHDR-E2E-SHDNet/trainsets/SD2VAU/test/Out \
        --output Output/unified_run/ \
        --weight Training_logs/Unified_R2MoE_SD2/models/80000_ema.pth \
        --save_mask \
        --save_grid
    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 infer_diffusion.py \
        --paradigm pcc \
        --input /data2/letrunglinh/despecular/diff-train-style/SHDR-E2E-SHDNet/trainsets/SD2VAU/test/Inp \
        --gt    /data2/letrunglinh/despecular/diff-train-style/SHDR-E2E-SHDNet/trainsets/SD2VAU/test/Out \
        --output Output/pcc_run/ \
        --weight_G Training_logs/AnyIR_SpecMamba_PCC_SD2/models/80000_E.pth \
        --weight_S Training_logs/AnyIR_SpecMamba_PCC_SD2/models/80000_SpecMamba.pth \
        --weight_t Training_logs/AnyIR_SpecMamba_PCC_SD2/models/80000_t_embed.pth \
        --anyir_dim 24 \
        --use_mask_film \
        --save_mask \                
        --save_grid           

    (flags must match the training config: --backbone anyir, --anyir_dim 48,
     --use_mask_film on, --predict_residual off, --specmamba_input_size 256, --T 8, --t_dim 128.)

Usage (legacy diffusion):
    torchrun --nproc_per_node=4 infer_diffusion.py \
        --paradigm diffusion --infer_steps 1 \
        --input ... --output ... \
        --weight_G .../70000_E.pth --weight_S .../70000_SpecMamba.pth --weight_t .../70000_t_embed.pth

Usage (multi-GPU, e.g. 4 GPUs):
    torchrun --nproc_per_node=4 infer_diffusion.py \
        --input ... --output ... \
        --weight_G ... --weight_S ... --weight_t ...

    # pick specific GPU indices (e.g. 0 and 2 only). Either of these works:
    CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 infer_diffusion.py ...
    torchrun --nproc_per_node=2 infer_diffusion.py --gpu_ids 0,2 ...

    # single-process on a specific GPU:
    python infer_diffusion.py --gpu_ids 3 ...

    # single image:
    python infer_diffusion.py --input path/to/img.png --output out_dir/ ...

    # with GT for PSNR/SSIM:
    python infer_diffusion.py --input ... --gt trainsets/SHIQ/Test/Out/ ...

    # use EMA weights (recommended for evaluation):
    python infer_diffusion.py --weight_G <step>_E.pth ...
"""

import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from models.Qformer import Qformer
from models.network_anyir import AnyIR
from models.specmamba import Specmamba
from models.diffusion_modules import (
    AlphaSchedule,
    DiffusionAnyIR,
    DiffusionQformer,
    DiffusionSpecMamba,
    TimeEmbedMLP,
    load_pretrained_into_wrapper,
)
from utils import utils_image as util


IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _extract_state(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    for k in ('params', 'model_state_dict', 'state_dict'):
        v = ckpt.get(k)
        if isinstance(v, dict):
            return v
    return ckpt


def _strip_module(sd):
    from collections import OrderedDict
    out = OrderedDict()
    for k, v in sd.items():
        out[k[7:] if k.startswith('module.') else k] = v
    return out


def load_into_wrapper(path, wrapper, device, label):
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(f'{label} checkpoint not found: {path}')
    print(f'Loading {label} <- {path}')
    ckpt = torch.load(path, map_location=device)
    sd = _strip_module(_extract_state(ckpt))
    load_pretrained_into_wrapper(wrapper, sd)


def build_models(args, device):
    if args.backbone == 'anyir':
        base_q = AnyIR(
            inp_channels=4,
            out_channels=4,
            dim=args.anyir_dim,
            num_blocks=args.anyir_num_blocks,
            num_refinement_blocks=args.anyir_num_refinement_blocks,
            heads=args.anyir_heads,
            ffn_expansion_factor=args.anyir_ffn_expansion_factor,
        )
        wrap_q = DiffusionAnyIR(
            base_q,
            t_dim=args.t_dim,
            use_mask_film=args.use_mask_film,
            predict_residual=args.predict_residual,
        ).to(device)
    else:
        base_q = Qformer()
        wrap_q = DiffusionQformer(
            base_q,
            t_dim=args.t_dim,
            use_mask_film=args.use_mask_film,
            predict_residual=args.predict_residual,
        ).to(device)

    base_s = Specmamba(
        base_dim=32, depths=(2, 2, 4, 4), shared_scan=True, prompt_dim=128,
        input_range='01', stem_stride=2, lite_expansion=1.0,
        use_mamba_decoder=False, use_mamba_stage3=True,
    )
    wrap_s = DiffusionSpecMamba(base_s, t_dim=args.t_dim).to(device)

    t_embed = TimeEmbedMLP(dim=args.t_dim).to(device)
    return wrap_q, wrap_s, t_embed


def load_t_embed(path, t_embed, device):
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(f't_embed checkpoint not found: {path}')
    print(f'Loading t_embed <- {path}')
    sd = torch.load(path, map_location=device)
    sd = _strip_module(_extract_state(sd))
    t_embed.load_state_dict(sd, strict=True)


# ---------------------------------------------------------------------------
# Image IO
# ---------------------------------------------------------------------------

def list_images(input_path):
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = []
        for ext in IMG_EXTS:
            files.extend(sorted(glob(str(p / f'*{ext}'))))
            files.extend(sorted(glob(str(p / f'*{ext.upper()}'))))
        return sorted(set(files))
    raise FileNotFoundError(f'Input path does not exist: {input_path}')


def read_image_tensor(path, device):
    """RGB image -> (1,3,H,W) float tensor in [0,1]."""
    img = util.imread_uint(path, 3)        # HWC uint8 RGB
    img = util.uint2single(img)            # HWC float [0,1]
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)
    return t.to(device, non_blocking=True)


def pad_to_multiple(x, multiple=8):
    """Right/bottom zero-pad so H and W are divisible by `multiple`."""
    _, _, h, w = x.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return x, (h, w)
    return F.pad(x, (0, pw, 0, ph), mode='reflect'), (h, w)


def save_tensor_image(t, path):
    """t: (1,3,H,W) or (3,H,W) or (1,1,H,W) or (1,H,W), values in [0,1]."""
    if t.dim() == 4:
        t = t[0]
    arr = t.detach().clamp(0, 1).float().cpu().numpy()
    if arr.shape[0] == 1:
        arr = arr[0]                      # (H,W)
    else:
        arr = arr.transpose(1, 2, 0)      # (H,W,3)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    Image.fromarray(arr).save(path)


# ---------------------------------------------------------------------------
# DDIM sampling
# ---------------------------------------------------------------------------

def _specmamba_size(specmamba_input_size, spatial_size):
    if specmamba_input_size is None:
        return spatial_size
    return (specmamba_input_size, specmamba_input_size)


@torch.no_grad()
def sample(L, wrap_q, wrap_s, t_embed, sched, T, steps, device, specmamba_input_size=None, return_trace=False):
    """Iterative DDIM sampling.

    Args:
        L     : (B,3,H,W) input image in [0,1].
        steps : number of denoising steps (<= T).
    Returns:
        H_hat : (B,3,H,W) clean image.
        M     : (B,1,H,W) final specular probability mask.
        trace : list of (H_hat, M) per step (only if return_trace).
    """
    B, _, H, W = L.shape
    steps = max(1, min(steps, T))
    ts = torch.linspace(T, 0, steps + 1, device=device).round().long()

    x = L.clone()
    H_hat = L.clone()
    spec = None
    trace = []

    for i in range(steps):
        t_now = ts[i].expand(B)
        t_next = ts[i + 1].expand(B)
        alpha_t = sched.alpha(t_now).view(-1, 1, 1, 1)
        alpha_n = sched.alpha(t_next).view(-1, 1, 1, 1)

        t_emb = t_embed(t_now)

        spec_size = _specmamba_size(specmamba_input_size, (H, W))
        x_spec = F.interpolate(x, size=spec_size, mode='bilinear', align_corners=False)
        m_logits_spec = wrap_s(x_spec, t_emb)
        m_logits = F.interpolate(m_logits_spec, size=(H, W), mode='bilinear', align_corners=False)
        spec = F.softmax(m_logits, dim=1)[:, 1:2]

        q_in = torch.cat([spec, x], dim=1)
        q_out = wrap_q(q_in, t_emb, mask=spec)
        H_hat = q_out[:, 1:]

        h_hat = (x - H_hat) / alpha_t.clamp_min(1e-6)
        x = H_hat + alpha_n * h_hat

        if return_trace:
            trace.append((H_hat.clone(), spec.clone()))

    return H_hat.clamp(0, 1), spec, trace


@torch.no_grad()
def sample_pcc(L, wrap_q, wrap_s, t_embed, T, device, specmamba_input_size=None):
    """PCC inference: a single restoration forward at a fixed time t=T
    (no DDIM sampling). Mirrors ModelPlain._test_pcc.

    Returns (H_hat (B,3,H,W) in [0,1], M (B,1,H,W) specular probability).
    """
    B, _, H, W = L.shape
    t = torch.full((B,), int(T), device=device, dtype=torch.long)
    t_emb = t_embed(t)

    spec_size = _specmamba_size(specmamba_input_size, (H, W))
    x_spec = F.interpolate(L, size=spec_size, mode='bilinear', align_corners=False)
    m_logits_spec = wrap_s(x_spec)                       # SpecMamba is time-agnostic
    m_logits = F.interpolate(m_logits_spec, size=(H, W), mode='bilinear', align_corners=False)
    spec = F.softmax(m_logits, dim=1)[:, 1:2]

    q_out = wrap_q(torch.cat([spec, L], dim=1), t_emb, mask=spec)
    H_hat = q_out[:, 1:]
    return H_hat.clamp(0, 1), spec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_distributed():
    """Detect torchrun env and initialize NCCL. Returns (rank, world_size, local_rank).
    Caller is responsible for picking the physical device id (see _resolve_device)."""
    if not any(k in os.environ for k in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE')):
        return 0, 1, 0
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def _parse_gpu_ids(s):
    """'0,2,3' -> [0, 2, 3] ; '' / None -> []."""
    if not s:
        return []
    return [int(x.strip()) for x in s.split(',') if x.strip() != '']


def _parse_int_list(s):
    return [int(x.strip()) for x in s.split(',') if x.strip() != '']


def _resolve_device(args, world_size, local_rank):
    """Pick the physical GPU id for this rank.

    Priority:
      1. --gpu_ids on the CLI (maps LOCAL_RANK -> ids[LOCAL_RANK])
      2. Otherwise LOCAL_RANK directly (matches CUDA_VISIBLE_DEVICES if set externally)
      3. Otherwise args.device (single-process fallback, e.g. 'cuda' or 'cpu')
    """
    ids = _parse_gpu_ids(args.gpu_ids)
    if ids:
        if world_size > 1 and len(ids) < world_size:
            raise ValueError(
                f'--gpu_ids has {len(ids)} entries but world_size={world_size}; '
                'provide at least one id per rank.'
            )
        device_id = ids[local_rank] if world_size > 1 else ids[0]
        torch.cuda.set_device(device_id)
        return f'cuda:{device_id}'
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        return f'cuda:{local_rank}'
    return args.device


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='image file or folder')
    p.add_argument('--output', required=True, help='output dir')
    p.add_argument('--gt', default=None, help='optional GT folder (same filenames as input) for PSNR/SSIM')
    p.add_argument('--weight_G', required=True, help='restoration checkpoint (G or E .pth)')
    p.add_argument('--weight_S', required=True, help='SpecMamba checkpoint')
    p.add_argument('--weight_t', required=True, help='t_embed checkpoint')
    p.add_argument('--paradigm', choices=['pcc', 'diffusion'], default='pcc',
                   help='pcc = single forward at t=T (current); diffusion = DDIM sampling (legacy)')
    p.add_argument('--backbone', choices=['anyir', 'qformer'], default='anyir',
                   help='restoration backbone used during training')
    p.add_argument('--anyir_dim', type=int, default=48)
    p.add_argument('--anyir_num_blocks', type=_parse_int_list, default=[3, 5, 5, 7])
    p.add_argument('--anyir_heads', type=_parse_int_list, default=[1, 2, 4, 8])
    p.add_argument('--anyir_ffn_expansion_factor', type=float, default=2.0)
    p.add_argument('--anyir_num_refinement_blocks', type=int, default=4)
    p.add_argument('--use_mask_film', action='store_true',
                   help='enable multi-scale MaskFiLM; must match the checkpoint')
    p.add_argument('--predict_residual', action='store_true',
                   help='enable residual-prediction wrapper mode; must match the checkpoint')
    p.add_argument('--specmamba_input_size', type=int, default=256,
                   help='SpecMamba inference size. Use 256 for current AnyIR training config.')
    p.add_argument('--T', type=int, default=8, help='diffusion total steps (must match training)')
    p.add_argument('--t_dim', type=int, default=128, help='time-embed dim (must match training)')
    p.add_argument('--infer_steps', type=int, default=8, help='DDIM steps to run (<= T)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                   help='fallback device when --gpu_ids is not set and not running under torchrun')
    p.add_argument('--gpu_ids', default='',
                   help='comma-separated physical GPU ids (e.g. "0,2,3"). With torchrun, ids[LOCAL_RANK] is used; '
                        'for single-process, ids[0] is used. Equivalent to setting CUDA_VISIBLE_DEVICES.')
    p.add_argument('--save_mask', action='store_true', help='save predicted mask alongside the result')
    p.add_argument('--save_grid', action='store_true', help='save input | mask | result triptych')
    p.add_argument('--pad_multiple', type=int, default=8,
                   help='pad input so H,W are multiples of this (Qformer has 4 down-stages)')
    args = p.parse_args()

    # ---- multi-GPU setup (no-op for single-process)
    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)
    device = _resolve_device(args, world_size, local_rank)
    if is_main:
        if world_size > 1:
            print(f'Distributed inference: rank={rank}, world_size={world_size}, device={device}')
        else:
            print(f'Single-process inference, device={device}')

    torch.backends.cudnn.benchmark = True

    # ---- build + load (every rank loads the same weights)
    wrap_q, wrap_s, t_embed = build_models(args, device=device)
    if is_main:
        load_into_wrapper(args.weight_G, wrap_q, device, args.backbone)
        load_into_wrapper(args.weight_S, wrap_s, device, 'SpecMamba')
        load_t_embed(args.weight_t, t_embed, device)
    else:
        # silence load prints on non-zero ranks
        ckpt_g = torch.load(args.weight_G, map_location=device)
        load_pretrained_into_wrapper(wrap_q, _strip_module(_extract_state(ckpt_g)))
        ckpt_s = torch.load(args.weight_S, map_location=device)
        load_pretrained_into_wrapper(wrap_s, _strip_module(_extract_state(ckpt_s)))
        t_embed.load_state_dict(_strip_module(_extract_state(torch.load(args.weight_t, map_location=device))), strict=True)
    wrap_q.eval(); wrap_s.eval(); t_embed.eval()

    sched = AlphaSchedule(T=args.T).to(device)

    # ---- discover inputs and shard across ranks
    inputs = list_images(args.input)
    assert inputs, f'No images found under {args.input}'
    my_inputs = inputs[rank::world_size]                    # interleaved shard
    if is_main:
        mode = ('PCC single-pass (t=T)' if args.paradigm == 'pcc'
                else f'{args.infer_steps}-step DDIM (T={args.T})')
        print(f'Found {len(inputs)} input images. {world_size} rank(s); '
              f'rank-0 will process {len(my_inputs)}. Running {mode}.')

    has_gt = args.gt is not None and os.path.isdir(args.gt)
    psnr_sum, ssim_sum, n_evaluated = 0.0, 0.0, 0

    pbar = tqdm(my_inputs, dynamic_ncols=True, disable=not is_main)
    for path in pbar:
        name = os.path.splitext(os.path.basename(path))[0]
        L = read_image_tensor(path, device)
        L_pad, (h_orig, w_orig) = pad_to_multiple(L, args.pad_multiple)

        if args.paradigm == 'pcc':
            H_hat_pad, M_pad = sample_pcc(
                L_pad, wrap_q, wrap_s, t_embed,
                T=args.T, device=device,
                specmamba_input_size=args.specmamba_input_size,
            )
        else:
            H_hat_pad, M_pad, _ = sample(
                L_pad, wrap_q, wrap_s, t_embed, sched,
                T=args.T, steps=args.infer_steps, device=device,
                specmamba_input_size=args.specmamba_input_size,
            )
        H_hat = H_hat_pad[..., :h_orig, :w_orig]
        M_out = M_pad[..., :h_orig, :w_orig]

        save_tensor_image(H_hat, os.path.join(args.output, 'removal', f'{name}.png'))
        if args.save_mask:
            save_tensor_image(M_out, os.path.join(args.output, 'mask', f'{name}.png'))
        if args.save_grid:
            grid = torch.cat([L, M_out.repeat(1, 3, 1, 1), H_hat], dim=-1)
            save_tensor_image(grid, os.path.join(args.output, 'grid', f'{name}.png'))

        if has_gt:
            gt_candidates = [os.path.join(args.gt, f'{name}{ext}') for ext in IMG_EXTS]
            gt_path = next((q for q in gt_candidates if os.path.isfile(q)), None)
            if gt_path is not None:
                gt_uint = util.imread_uint(gt_path, 3)
                pr_uint = (H_hat[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
                if pr_uint.shape == gt_uint.shape:
                    psnr = util.calculate_psnr(pr_uint, gt_uint, border=0, test_y_channel=False)
                    ssim = util.calculate_ssim(pr_uint, gt_uint, border=0, scale=1)
                    psnr_sum += psnr
                    ssim_sum += ssim
                    n_evaluated += 1
                    if is_main:
                        pbar.set_postfix({'PSNR': f'{psnr:.2f}', 'avg': f'{psnr_sum/n_evaluated:.2f}'})

    # ---- all-reduce metrics across ranks
    if has_gt:
        stats = torch.tensor([psnr_sum, ssim_sum, float(n_evaluated)], device=device)
        if world_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if is_main:
            n_total = int(stats[2].item())
            if n_total > 0:
                print(f'\nEvaluated on {n_total} images vs GT:')
                print(f'  Avg PSNR: {stats[0].item() / n_total:.3f} dB')
                print(f'  Avg SSIM: {stats[1].item() / n_total:.4f}')

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
