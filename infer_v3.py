"""
Inference for Unified_R2MoE_SD2_v3.

Key v3 differences vs infer_unified.py:
  - deep_removal=True, removal_blocks=6  (RegimeGatedRemovalMoE full-res decoder)
  - defaults point to v3 checkpoint and SD2VAU test set

Usage (single GPU):
    python infer_v3.py \
        --input  /data2/letrunglinh/despecular/diff-train-style/SHDR-E2E-SHDNet/trainsets/SD2VAU/test/Inp \
        --gt     /data2/letrunglinh/despecular/diff-train-style/SHDR-E2E-SHDNet/trainsets/SD2VAU/test/Out \
        --output results/Unified_R2MoE_SD2_v3/ \
        --weight Training_logs/Unified_R2MoE_SD2_v3/models/best_ema.pth \
        --save_mask --save_grid --device cuda:2

Usage (multi-GPU via torchrun):
    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=29501 infer_v3.py \
        --input  /data2/letrunglinh/despecular/diff-train-style/SHDR-E2E-SHDNet/trainsets/SD2VAU/test/Inp \
        --gt     /data2/letrunglinh/despecular/diff-train-style/SHDR-E2E-SHDNet/trainsets/SD2VAU/test/Out \
        --output results/Unified_R2MoE_SD2_v3/ \
        --weight Training_logs/Unified_R2MoE_SD2_v3/models/best_ema.pth \
        --save_mask

    pyiqa psnr -t /data2/letrunglinh/despecular/diff-train-style/SHR-specmamba-anyir/results/Unified_R2MoE_SD2_v3/removal -r /data2/letrunglinh/despecular/diff-train-style/SHDR-E2E-SHDNet/trainsets/SD2VAU/test/Out  --device cuda --verbose
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

from models.specmamba import Specmamba
from utils import utils_image as util


IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


# ---------------------------------------------------------------------------
# Helpers (identical to infer_unified.py)
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
    img = util.imread_uint(path, 3)
    img = util.uint2single(img)
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)
    return t.to(device, non_blocking=True)


def pad_to_multiple(x, multiple=8):
    _, _, h, w = x.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return x, (h, w)
    return F.pad(x, (0, pw, 0, ph), mode='reflect'), (h, w)


def save_tensor_image(t, path):
    if t.dim() == 4:
        t = t[0]
    arr = t.detach().clamp(0, 1).float().cpu().numpy()
    if arr.shape[0] == 1:
        arr = arr[0]
    else:
        arr = arr.transpose(1, 2, 0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    Image.fromarray(arr).save(path)


def setup_distributed():
    if not any(k in os.environ for k in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE')):
        return 0, 1, 0
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    return rank, world_size, local_rank


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Infer with Unified_R2MoE_SD2_v3')
    p.add_argument('--input',   default='/data2/letrunglinh/despecular/diff-train-style/SHDR-E2E-SHDNet/trainsets/SD2VAU/test/Inp',
                   help='image file or folder')
    p.add_argument('--output',  default='results/Unified_R2MoE_SD2_v3/',
                   help='output directory')
    p.add_argument('--gt',      default='/data2/letrunglinh/despecular/diff-train-style/SHDR-E2E-SHDNet/trainsets/SD2VAU/test/Out',
                   help='GT folder for PSNR/SSIM; pass empty string to skip')
    p.add_argument('--weight',  default='Training_logs/Unified_R2MoE_SD2_v3/models/best_ema.pth',
                   help='checkpoint path')
    p.add_argument('--save_mask',  action='store_true', help='save predicted specular mask')
    p.add_argument('--save_grid',  action='store_true', help='save input | mask | result triptych')
    p.add_argument('--device',     default='cuda:2',   help='device for single-GPU mode')
    p.add_argument('--max_imgs',   type=int, default=0, help='cap number of images (0 = all)')
    # v3 architecture (match train_unified_SD2_v3.json)
    p.add_argument('--base_dim',          type=int,   default=32)
    p.add_argument('--depths',            type=str,   default='2,2,4,4')
    p.add_argument('--moe_experts',       type=int,   default=4)
    p.add_argument('--prompt_dim',        type=int,   default=128)
    p.add_argument('--stem_stride',       type=int,   default=2)
    p.add_argument('--lite_expansion',    type=float, default=1.0)
    p.add_argument('--removal_blocks',    type=int,   default=6)
    p.add_argument('--removal_dim',       type=int,   default=0,
                   help='width of the removal reconstruction path (0 = same as base_dim). Must match training.')
    p.add_argument('--no_deep_removal',   action='store_true', help='disable deep_removal (not recommended for v3)')
    # NOTE: train_unified_SD2_v3.json sets use_mamba_decoder=true -> dec3 is a SpecMambaBlock.
    # This MUST match training or dec3 weights are silently dropped (strict=False) and run random-init.
    p.add_argument('--use_mamba_decoder', dest='use_mamba_decoder', action='store_true', default=True,
                   help='dec3 = SpecMambaBlock (matches v3 config; default ON)')
    p.add_argument('--no_mamba_decoder',  dest='use_mamba_decoder', action='store_false',
                   help='dec3 = LiteStage (only if you trained with use_mamba_decoder=false)')
    p.add_argument('--pad_multiple',      type=int,   default=8)
    args = p.parse_args()

    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        device = args.device if torch.cuda.is_available() else 'cpu'

    torch.backends.cudnn.benchmark = True

    # ---- build v3 model
    depths = tuple(int(x) for x in args.depths.split(','))
    net = Specmamba(
        n_channels=3,
        n_classes=2,
        base_dim=args.base_dim,
        depths=depths,
        shared_scan=True,
        prompt_dim=args.prompt_dim,
        input_range='01',
        stem_stride=args.stem_stride,
        lite_expansion=args.lite_expansion,
        use_mamba_decoder=args.use_mamba_decoder,
        use_mamba_stage3=True,
        dual_head=True,
        moe_experts=args.moe_experts,
        deep_removal=not args.no_deep_removal,
        removal_blocks=args.removal_blocks,
        removal_dim=(args.removal_dim or None),
    ).to(device)

    # ---- load weights
    if not os.path.isfile(args.weight):
        raise FileNotFoundError(f'Checkpoint not found: {args.weight}')
    if is_main:
        print(f'Loading weights <- {args.weight}')
    ckpt = torch.load(args.weight, map_location=device)
    sd = _strip_module(_extract_state(ckpt))
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if is_main:
        n_params = sum(p.numel() for p in net.parameters()) / 1e6
        print(f'  Params: {n_params:.2f}M  |  missing={len(missing)}  unexpected={len(unexpected)}')
        if missing or unexpected:
            print('  *** ARCHITECTURE MISMATCH — loaded weights do NOT match this model. ***')
            print('  *** Inference results will be WRONG. Check --use_mamba_decoder / --depths / flags. ***')
            if missing:
                print(f'  Missing keys (first 5): {missing[:5]}')
            if unexpected:
                print(f'  Unexpected keys (first 5): {unexpected[:5]}')
    net.eval()

    # ---- discover + shard inputs
    inputs = list_images(args.input)
    assert inputs, f'No images found: {args.input}'
    if args.max_imgs > 0:
        inputs = inputs[:args.max_imgs]
    my_inputs = inputs[rank::world_size]
    if is_main:
        print(f'Found {len(inputs)} images. {world_size} rank(s). Running v3 R²-MoE inference.')

    has_gt = bool(args.gt) and os.path.isdir(args.gt)
    psnr_sum = ssim_sum = 0.0
    n_eval = 0

    pbar = tqdm(my_inputs, dynamic_ncols=True, disable=not is_main)
    with torch.no_grad():
        for path in pbar:
            name = os.path.splitext(os.path.basename(path))[0]
            L = read_image_tensor(path, device)
            L_pad, (h0, w0) = pad_to_multiple(L, args.pad_multiple)

            mask_logits_pad, D_hat_pad = net(L_pad, return_aux=False)
            D_hat    = D_hat_pad[..., :h0, :w0].clamp(0, 1)
            spec_prob = F.softmax(mask_logits_pad.float(), dim=1)[:, 1:2][..., :h0, :w0]

            save_tensor_image(D_hat, os.path.join(args.output, 'removal', f'{name}.png'))
            if args.save_mask:
                save_tensor_image(spec_prob, os.path.join(args.output, 'mask', f'{name}.png'))
            if args.save_grid:
                grid = torch.cat([L[..., :h0, :w0],
                                  spec_prob.repeat(1, 3, 1, 1),
                                  D_hat], dim=-1)
                save_tensor_image(grid, os.path.join(args.output, 'grid', f'{name}.png'))

            if has_gt:
                gt_candidates = [os.path.join(args.gt, f'{name}{ext}') for ext in IMG_EXTS]
                gt_path = next((q for q in gt_candidates if os.path.isfile(q)), None)
                if gt_path is not None:
                    gt_u = util.imread_uint(gt_path, 3)
                    pr_u = (D_hat[0].cpu().permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
                    if pr_u.shape == gt_u.shape:
                        psnr = util.calculate_psnr(pr_u, gt_u, border=0, test_y_channel=False)
                        ssim = util.calculate_ssim(pr_u, gt_u, border=0, scale=1)
                        psnr_sum += psnr
                        ssim_sum += ssim
                        n_eval   += 1
                        if is_main:
                            pbar.set_postfix({'PSNR': f'{psnr:.2f}', 'avg': f'{psnr_sum/n_eval:.2f}'})

    if has_gt:
        stats = torch.tensor([psnr_sum, ssim_sum, float(n_eval)], device=device)
        if world_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if is_main:
            n = int(stats[2].item())
            if n > 0:
                avg_psnr = stats[0].item() / n
                avg_ssim = stats[1].item() / n
                print(f'\nEvaluated on {n} images vs GT:')
                print(f'  Avg PSNR : {avg_psnr:.4f} dB')
                print(f'  Avg SSIM : {avg_ssim:.4f}')
                log_path = os.path.join(args.output, 'metrics.txt')
                os.makedirs(args.output, exist_ok=True)
                with open(log_path, 'w') as f:
                    f.write(f'ckpt={args.weight}\n')
                    f.write(f'n={n}  PSNR={avg_psnr:.4f}  SSIM={avg_ssim:.4f}\n')
                print(f'  Saved -> {log_path}')

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
