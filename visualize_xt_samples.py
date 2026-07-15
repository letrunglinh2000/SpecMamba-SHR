"""
Save x_t samples at each diffusion timestep for paper figures.

Shows:  L (input)  |  x_t=T  x_t=T-1  ...  x_t=1  |  H (GT clean)
Also optionally overlays SpecMamba mask predictions if weights are provided.

Usage:
    python visualize_xt_samples.py --opt options/train_anyir_SD1.json
    python visualize_xt_samples.py --opt options/train_anyir.json --n_samples 4
    python visualize_xt_samples.py --opt options/train_anyir_SD1.json \
        --weightSpecMamba weight/SpecMamba/SD2_SpecMamba.pth --show_mask
    conda run -n linh python visualize_xt_samples.py \
        --opt options/train_anyir_SD1.json \
        --weightSpecMamba /data2/letrunglinh/despecular/diff-train-style/SHR-specmamba-anyir/Training_logs/AnyIR_SpecMamba_Diffusion/models/55000_SpecMamba.pth \
        --show_mask --n_samples 4
    
    conda run -n linh python visualize_xt_samples.py \
    --opt options/train_anyir.json \
    --weightSpecMamba /data2/letrunglinh/despecular/diff-train-style/SHR-specmamba-anyir/weight/SHIQ_200000_SpecMamba.pth \
    --show_mask --n_samples 4
"""
import argparse
import os
import sys
import json
import re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── helpers ──────────────────────────────────────────────────────────────────

def strip_json_comments(text):
    text = re.sub(r'//[^\n]*', '', text)
    return text

def load_json(path):
    with open(path) as f:
        return json.loads(strip_json_comments(f.read()))

def load_img(path):
    """Load image as float32 numpy HWC [0,1]."""
    img = np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0
    return img

def img_to_tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

def tensor_to_img(t):
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()

def cosine_alpha(t, T):
    return 0.5 * (1.0 - np.cos(np.pi * t / T))

def center_crop(img, size=256):
    h, w = img.shape[:2]
    y = (h - size) // 2
    x = (w - size) // 2
    y, x = max(0, y), max(0, x)
    return img[y:y+size, x:x+size]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', default='options/train_anyir_SD1.json')
    parser.add_argument('--n_samples', type=int, default=4,
                        help='Number of image pairs to visualize')
    parser.add_argument('--sample_ids', type=str, default=None,
                        help='Comma-separated indices, e.g. 0,5,10,20')
    parser.add_argument('--crop', type=int, default=256)
    parser.add_argument('--weightSpecMamba', type=str, default=None)
    parser.add_argument('--show_mask', action='store_true')
    parser.add_argument('--inp_dir', type=str, default=None,
                        help='Override dataroot_L from opt (use when opt has relative paths)')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Override dataroot_H from opt')
    parser.add_argument('--out_dir', default='xt_samples')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    opt = load_json(args.opt)
    T = opt['train']['diffusion']['T']
    mask_thresh = opt['train']['diffusion']['mask_alpha_thresh']

    # dataset paths — command-line overrides opt file (needed when opt has relative paths)
    d = opt['datasets']['train']
    inp_dir = Path(args.inp_dir) if args.inp_dir else Path(d['dataroot_L'])
    gt_dir  = Path(args.gt_dir)  if args.gt_dir  else Path(d['dataroot_H'])
    all_files = sorted(inp_dir.glob('*'))

    # pick sample indices
    if args.sample_ids:
        indices = [int(i) for i in args.sample_ids.split(',')]
    else:
        step = max(1, len(all_files) // args.n_samples)
        indices = list(range(0, len(all_files), step))[:args.n_samples]

    print(f"Dataset: {inp_dir}  ({len(all_files)} images)")
    print(f"T={T},  samples: {indices}")

    # optionally load SpecMamba — must match training architecture exactly
    spec_model = None
    if args.show_mask and args.weightSpecMamba:
        sys.path.insert(0, str(Path(__file__).parent))
        from models.specmamba import Specmamba
        spec_model = Specmamba(
            base_dim=32,
            depths=(2, 2, 4, 4),
            shared_scan=True,
            prompt_dim=128,
            input_range="01",
            stem_stride=2,
            lite_expansion=1.0,
            use_mamba_decoder=False,
            use_mamba_stage3=True,
        ).to(args.device)
        ckpt = torch.load(args.weightSpecMamba, map_location='cpu')
        # handle multiple checkpoint formats:
        #   - DiffusionSpecMamba (training): keys have 'base.' prefix, saved under 'params'
        #   - pretrained raw Specmamba:       no prefix, saved under 'model_state_dict'
        #   - DDP wrapped:                    'module.' prefix
        if isinstance(ckpt, dict):
            state = (ckpt.get('model_state_dict') or
                     ckpt.get('params') or
                     ckpt.get('state_dict') or
                     ckpt)
        else:
            state = ckpt
        def clean_keys(sd):
            out = {}
            for k, v in sd.items():
                k = k[len('module.'):] if k.startswith('module.') else k
                k = k[len('base.'):] if k.startswith('base.') else k
                out[k] = v
            return out
        state = clean_keys(state)
        missing, unexpected = spec_model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:3]} ...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")
        spec_model.eval()
        print(f"Loaded SpecMamba from {args.weightSpecMamba}")

    # ── per-sample figure ────────────────────────────────────────────────────
    for idx in indices:
        fpath = all_files[idx]
        gt_path = gt_dir / fpath.name
        if not gt_path.exists():
            print(f"  GT not found for {fpath.name}, skipping")
            continue

        L_np = center_crop(load_img(fpath),   args.crop)
        H_np = center_crop(load_img(gt_path), args.crop)
        L = img_to_tensor(L_np)
        H = img_to_tensor(H_np)
        h_gt = (L - H).clamp(0)  # specular residual

        # compute x_t for each timestep
        t_show = list(range(T, -1, -1))     # T, T-1, ..., 0
        xt_imgs = []
        alphas  = []
        for t in t_show:
            a = cosine_alpha(t, T)
            alphas.append(a)
            xt = H + a * (L - H)
            xt_imgs.append(tensor_to_img(xt))

        # pre-compute soft pseudo-mask from h_gt = (L-H).clamp(0) — supervision target
        # normalize per-image so the colormap uses the full range
        h_soft_base = h_gt.max(dim=1, keepdim=True)[0]          # (1,1,H,W) max across RGB
        h_max = h_soft_base.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        h_soft_norm = (h_soft_base / h_max).squeeze().cpu().numpy()  # (H,W) in [0,1]

        # ── build figure ─────────────────────────────────────────────────────
        n_cols = len(t_show) + 2    # L | x_T ... x_0 | H
        # rows: 0=images, 1=soft mask, 2=SpecMamba (optional)
        n_rows = 2 + (1 if spec_model is not None else 0)

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(2.2 * n_cols, 2.5 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        def show(ax, img, title, border_color=None):
            ax.imshow(img)
            ax.set_title(title, fontsize=8)
            ax.axis('off')
            if border_color:
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)
                    spine.set_visible(True)

        # row 0: x_t images
        show(axes[0, 0], L_np, 'L (input)', border_color='darkorange')
        for col, (t, a, img) in enumerate(zip(t_show, alphas, xt_imgs), start=1):
            active = a > mask_thresh
            bc = 'mediumpurple' if active else 'lightgray'
            label = f'x_t  t={t}\nα={a:.3f}'
            if t == T:
                label += '\n(= L)'
            elif t == 0:
                label += '\n(≈ H)'
            show(axes[0, col], img, label, border_color=bc)
        show(axes[0, -1], H_np, 'H (GT clean)', border_color='green')

        # row 1: soft pseudo-mask from (x_t - H).clamp(0) — scales with alpha_t
        axes[1, 0].axis('off')
        axes[1, 0].text(0.5, 0.5, 'soft mask\n(x_t − H)', ha='center', va='center',
                        transform=axes[1, 0].transAxes, fontsize=9)
        for col, (t, a) in enumerate(zip(t_show, alphas), start=1):
            soft_t = (a * h_soft_norm).clip(0, 1)
            axes[1, col].imshow(soft_t, cmap='hot', vmin=0, vmax=1)
            axes[1, col].set_title(f'soft t={t}\nα={a:.2f}', fontsize=7)
            axes[1, col].axis('off')
        axes[1, -1].axis('off')

        # row 2: SpecMamba predicted mask (optional)
        if spec_model is not None:
            axes[2, 0].axis('off')
            axes[2, 0].text(0.5, 0.5, 'SpecMamba\nsoft mask', ha='center', va='center',
                            transform=axes[2, 0].transAxes, fontsize=9)
            for col, (t, a, xt_img) in enumerate(zip(t_show, alphas, xt_imgs), start=1):
                xt_t = torch.from_numpy(xt_img).permute(2, 0, 1).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    logits = spec_model(xt_t)
                prob = logits.softmax(dim=1)[0, 1].cpu().numpy()
                axes[2, col].imshow(prob, cmap='hot', vmin=0, vmax=1)
                axes[2, col].set_title(f'mask t={t}', fontsize=7)
                axes[2, col].axis('off')
            axes[2, -1].axis('off')

        # diff map (specular residual)
        diff_vis = (h_gt * 3).clamp(0, 1)  # ×3 for visibility
        diff_vis = tensor_to_img(diff_vis)

        fig.suptitle(f'Sample: {fpath.name}  |  purple border = mask loss active (α > {mask_thresh})',
                     fontsize=10, fontweight='bold')
        plt.tight_layout()

        out_path = os.path.join(args.out_dir, f'xt_{fpath.stem}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_path}")

    # ── summary grid (all samples, t=T, t=T//2, t=1, H) ─────────────────────
    t_select = [T, T // 2, 1, 0]
    a_select = [cosine_alpha(t, T) for t in t_select]
    col_labels = [f't={t} α={a:.2f}' for t, a in zip(t_select, a_select)]
    col_labels = ['L (input)'] + col_labels + ['H (GT)']

    fig2, axes2 = plt.subplots(len(indices), len(col_labels),
                                figsize=(2.2 * len(col_labels), 2.5 * len(indices)))
    if len(indices) == 1:
        axes2 = axes2[np.newaxis, :]

    for row, idx in enumerate(indices):
        fpath = all_files[idx]
        gt_path = gt_dir / fpath.name
        if not gt_path.exists():
            continue
        L_np = center_crop(load_img(fpath),   args.crop)
        H_np = center_crop(load_img(gt_path), args.crop)
        L = img_to_tensor(L_np)
        H = img_to_tensor(H_np)

        axes2[row, 0].imshow(L_np); axes2[row, 0].axis('off')
        for c, t in enumerate(t_select, start=1):
            a = cosine_alpha(t, T)
            xt = H + a * (L - H)
            axes2[row, c].imshow(tensor_to_img(xt)); axes2[row, c].axis('off')
        axes2[row, -1].imshow(H_np); axes2[row, -1].axis('off')

        if row == 0:
            for c, lbl in enumerate(col_labels):
                axes2[row, c].set_title(lbl, fontsize=9)

    fig2.suptitle(f'Diffusion interpolation T={T}  (summary grid)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    summary_path = os.path.join(args.out_dir, 'summary_grid.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"\nSummary grid saved to {summary_path}")


if __name__ == '__main__':
    main()
