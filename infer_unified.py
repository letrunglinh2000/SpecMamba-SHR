"""
Inference for the unified R2-MoE Specmamba (joint detection + removal).

The model is a single Specmamba(dual_head=True) that outputs:
    mask_logits  (B,2,H,W)  -- specular probability map
    D_hat        (B,3,H,W)  -- diffuse (de-specularised) image

Usage (full test set, 4 GPUs, best checkpoint):
    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=29501 infer_unified.py \
        --input  /data2/letrunglinh/despecular/diff-train-style/SHDR-E2E-SHDNet/trainsets/SD2VAU/test/Inp \
        --gt     /data2/letrunglinh/despecular/diff-train-style/SHDR-E2E-SHDNet/trainsets/SD2VAU/test/Out \
        --output Output/unified_run/ \
        --weight Training_logs/Unified_R2MoE_SD2_v3/models/best_ema.pth \
        --save_mask \
        --save_grid

Usage (single GPU, specific checkpoint):
    python infer_unified.py \
        --input  .../test/Inp \
        --output Output/unified_run/ \
        --weight Training_logs/Unified_R2MoE_SD2/models/55000_ema.pth
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
# Helpers
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
    p = argparse.ArgumentParser()
    p.add_argument('--input',  required=True, help='image file or folder')
    p.add_argument('--output', required=True, help='output directory')
    p.add_argument('--gt',     default=None,  help='optional GT folder for PSNR/SSIM')
    p.add_argument('--weight', required=True, help='path to _ema.pth or _net.pth checkpoint')
    p.add_argument('--save_mask', action='store_true', help='also save predicted specular mask')
    p.add_argument('--save_grid', action='store_true', help='save input | mask | result triptych')
    # model architecture (must match training config)
    p.add_argument('--base_dim',          type=int,   default=32)
    p.add_argument('--depths',            type=str,   default='2,2,4,4')
    p.add_argument('--moe_experts',       type=int,   default=4)
    p.add_argument('--prompt_dim',        type=int,   default=128)
    p.add_argument('--stem_stride',       type=int,   default=2)
    p.add_argument('--lite_expansion',    type=float, default=1.0)
    p.add_argument('--use_mamba_decoder', action='store_true')
    p.add_argument('--pad_multiple',      type=int,   default=8)
    args = p.parse_args()

    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.backends.cudnn.benchmark = True

    # ---- build model
    depths = tuple(int(x) for x in args.depths.split(','))
    net = Specmamba(
        n_channels=3, n_classes=2,
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
        print(f'  Loaded: {len(sd)-len(unexpected)} keys | missing={len(missing)} | unexpected={len(unexpected)}')
    net.eval()

    # ---- discover + shard inputs
    inputs = list_images(args.input)
    assert inputs, f'No images found: {args.input}'
    my_inputs = inputs[rank::world_size]
    if is_main:
        print(f'Found {len(inputs)} images. {world_size} rank(s). Running unified R2-MoE inference.')

    has_gt = args.gt is not None and os.path.isdir(args.gt)
    psnr_sum = ssim_sum = 0.0
    n_eval = 0

    pbar = tqdm(my_inputs, dynamic_ncols=True, disable=not is_main)
    with torch.no_grad():
        for path in pbar:
            name = os.path.splitext(os.path.basename(path))[0]
            L = read_image_tensor(path, device)
            L_pad, (h0, w0) = pad_to_multiple(L, args.pad_multiple)

            mask_logits_pad, D_hat_pad = net(L_pad, return_aux=False)
            D_hat = D_hat_pad[..., :h0, :w0].clamp(0, 1)
            spec_prob = F.softmax(mask_logits_pad, dim=1)[:, 1:2][..., :h0, :w0]

            save_tensor_image(D_hat, os.path.join(args.output, 'removal', f'{name}.png'))
            if args.save_mask:
                save_tensor_image(spec_prob, os.path.join(args.output, 'mask', f'{name}.png'))
            if args.save_grid:
                grid = torch.cat([L, spec_prob.repeat(1, 3, 1, 1), D_hat], dim=-1)
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
                        psnr_sum += psnr; ssim_sum += ssim; n_eval += 1
                        if is_main:
                            pbar.set_postfix({'PSNR': f'{psnr:.2f}', 'avg': f'{psnr_sum/n_eval:.2f}'})

    if has_gt:
        stats = torch.tensor([psnr_sum, ssim_sum, float(n_eval)], device=device)
        if world_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if is_main:
            n = int(stats[2].item())
            if n > 0:
                print(f'\nEvaluated on {n} images vs GT:')
                print(f'  Avg PSNR : {stats[0].item()/n:.3f} dB')
                print(f'  Avg SSIM : {stats[1].item()/n:.4f}')

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
