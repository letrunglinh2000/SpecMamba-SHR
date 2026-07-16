# SpecMamba — Training on a New Dataset

This guide walks through training the unified SpecMamba (R2MoE) model on your own
specular-highlight-removal dataset. It covers the data layout, the config file you
edit, launching training, resuming, and running inference.

---

## 1. Prepare your dataset

The model trains on **aligned triplets**: input (specular), ground-truth (clean), and
an optional binary specular mask. Put them in three parallel folders:

```
<your_dataset>/
  train/
    Inp/     # L — input images WITH specular highlights
    Out/     # H — ground-truth clean images
    Mask/    # M — binary specular masks (optional, see §5)
  test/
    Inp/
    Out/
    Mask/    # optional; test mask is usually null
```

Requirements (enforced by `data/dataset_specular.py`):

- **Same count** in `Inp/`, `Out/`, and `Mask/`. The loader zips them by sorted
  filename order, so **filenames must sort into matching triplets** (e.g. `0001.png`
  in all three folders).
- For the **train** split, each triplet must be **pixel-aligned and the same H×W**.
  Size-mismatched pairs are auto-detected via the image header and skipped (you'll see
  `[DatasetSpecular] skipped N size-mismatched/unreadable pairs`). This is fine for a
  few outliers but a symptom of misalignment if many are dropped.
- Masks are binarized at load time with threshold `0.5` (`img_M >= 0.5 -> 1`). Any
  grayscale mask works; white = specular region.
- Images are RGB (`n_channels: 3`). Random 256×256 crops + flips/rotations are applied
  during training (`H_size`), so source images should be ≥ `H_size`.

---

## 2. Create a config

Copy an existing config and edit the paths. The v4 recipe is the current best baseline:

```bash
cp options/train_unified_SD2_v4.json options/train_unified_MYDATA.json
```

Edit these fields in `options/train_unified_MYDATA.json`:

```jsonc
{
  "task": "Unified_R2MoE_MYDATA"      // output dir name under Training_logs/
  , "gpu_ids": [0, 1, 2, 3]           // GPUs to use (must match --nproc_per_node)

  , "datasets": {
    "train": {
      "dataset_type": "spec"          // keep as "spec" (DatasetSpecular)
      , "dataroot_H": "/abs/path/<your_dataset>/train/Out/"
      , "dataroot_L": "/abs/path/<your_dataset>/train/Inp/"
      , "dataroot_M": "/abs/path/<your_dataset>/train/Mask/"   // or null (see §5)
      , "H_size": 256
      , "dataloader_batch_size": 16   // PER-GPU batch size
      , "dataloader_num_workers": 4
    }
    , "test": {
      "dataset_type": "spec"
      , "dataroot_H": "/abs/path/<your_dataset>/test/Out/"
      , "dataroot_L": "/abs/path/<your_dataset>/test/Inp/"
      , "dataroot_M": null            // test-time masks are predicted, not needed
      , "H_size": 256
    }
  }
}
```

Leave `model_args`, `train.lr`, losses, etc. at their defaults for a first run — they
are the tuned v4 recipe (see §7 for what each knob does).

> **Paths are absolute.** Use full paths; the trainer does not resolve relative dataroots
> against the repo.

---

## 3. Launch training

Use `torchrun` with one process per GPU. `--nproc_per_node` must equal the number of
GPUs in `gpu_ids`.

```bash
cd /data2/letrunglinh/despecular/diff-train-style/SHR-specmamba-anyir

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    main_train_forall.py --opt options/train_unified_MYDATA.json
```

Notes:

- Distributed mode auto-enables when `torchrun` env vars are present — no `--dist` flag
  needed.
- If you run two jobs on one machine, give the second a distinct port:
  `torchrun --nproc_per_node=4 --master_port=29503 ...`.
- Single-GPU is supported: `CUDA_VISIBLE_DEVICES=0 python main_train_forall.py --opt ...`.

Everything is written to `Training_logs/<task>/`:

```
Training_logs/Unified_R2MoE_MYDATA/
  train.log              # loss + eval history
  models/                # checkpoints: <step>_G.pth, <step>_SpecMamba.pth, best_*.pth, best_ema.pth
  images/                # eval visualizations (if save_images: true)
  options/               # the resolved config used for this run
```

---

## 4. Reading the logs, checkpoints, and eval

Training prints every `checkpoint_print` iters:

```
<iter: 50,000, lr:8.58e-05> loss:6.17e-02 L_mask:3.24e-02 L_rem:1.46e-02
  (char:8.07e-03 fft:6.79e-03 ssim:2.93e-02) L_bal:1.29e-02 ...
```

- `L_mask` — specular-mask prediction loss
- `L_rem` — removal (restoration) loss = `char` + `fft` + `ssim` terms
- `L_bal` — MoE load-balancing loss

Every `checkpoint_test` iters it runs eval on the **rank-0 GPU** over the test set and
logs PSNR/SSIM:

```
Eval (n=1700): PSNR=32.46dB SSIM=0.9662  (best=32.67@45,000)
```

- Models save every `checkpoint_save` iters. The **best-PSNR** checkpoint is tracked in
  `models/best_metric.json` and saved as `best_*.pth` (and `best_ema.pth`) — a resume
  won't clobber it with a worse model.
- Eval can take 10–15 min on a large test set; the other ranks wait at a barrier (the
  NCCL watchdog timeout is raised to 2 h so it won't false-fire).

---

## 5. Datasets with NO ground-truth masks

If your dataset has no specular masks, the model can still learn removal — it just won't
get direct mask supervision:

1. Set `"dataroot_M": null` in the **train** dataset. The loader feeds zero masks and
   `has_mask=0`.
2. Set `"lam_mask": 0.0` in `train` so the (meaningless) mask loss doesn't contribute.
3. **Recommended:** warm-start the SpecMamba prior from a mask-trained checkpoint and
   freeze it, so it still provides a useful specular prior:
   ```jsonc
   "path": { "pretrained_netSpecMamba": "weight/SpecMamba/SHIQ_SpecMamba.pth" }
   , "train": { "specmamba_freeze_warmup": 999999 }   // keep the prior frozen
   ```

This treats SpecMamba as a **reusable, dataset-agnostic specular prior** and trains only
the restoration backbone on your data.

---

## 6. Resuming training

Resume is **automatic**. Re-run the exact same `torchrun` command with the same config —
`main_train_forall.py` finds the latest `<step>_*.pth` in `models/`, restores the model,
optimizer, EMA, and best-metric state, and continues from that step. To train from
scratch, point `task` at a fresh (empty) directory or clear `models/`.

To **warm-start** from another model's weights (different run, not a resume), set
`path.pretrained` (or the per-net `pretrained_netG` / `pretrained_netSpecMamba` /
`pretrained_t_embed`) to a checkpoint path. Loading is `strict=False`, so changed heads
are skipped — verify the architecture matches before relying on this, or weights get
dropped silently.

---

## 7. Key knobs (defaults are the tuned v4 recipe)

`model_args` — **must match at inference time** (see §8):

| key | default | meaning |
|-----|---------|---------|
| `base_dim` | 32 | backbone width |
| `depths` | `[2,2,4,4]` | blocks per stage |
| `moe_experts` | 4 | number of removal experts |
| `removal_dim` | 128 | removal-head width — the main PSNR/capacity lever |
| `removal_blocks` | 6 | depth of the removal head |
| `use_mamba_decoder` | true | Mamba blocks in the decoder |
| `prompt_dim` | 128 | prompt/conditioning width |

`train`:

| key | default | meaning |
|-----|---------|---------|
| `lr` / `scheduler` / `warmup_iters` | 1e-4 / cosine / 1000 | optimizer schedule |
| `total_iters` | 200000 | total training steps |
| `lam_mask` / `lam_removal` / `lam_bal` | 1.0 / 2.0 / 0.01 | loss weights |
| `rem_w_fft` / `rem_w_ssim` | 0.1 / 0.2 | FFT & SSIM sub-loss weights |
| `use_amp` / `amp_dtype` | true / `bf16` | mixed precision (bf16 recommended) |
| `checkpoint_save` / `_test` / `_print` | 5000 / 5000 / 100 | save/eval/print cadence |
| `manual_seed` | 42 | reproducibility |

For high-coverage masks, keep cycle/sparse losses off (they're already 0 here). Use
`bf16` — it's the stable AMP dtype for this model.

---

## 8. Inference / evaluation

Use `infer_v4.py`. **The architecture flags must match your training `model_args`** —
otherwise weights are silently dropped by `strict=False` and results are wrong. The
`infer_v4.py` defaults already match `train_unified_SD2_v4.json`; override only what you
changed.

Single GPU:

```bash
python infer_v4.py \
    --input  /abs/path/<your_dataset>/test/Inp/ \
    --gt     /abs/path/<your_dataset>/test/Out/ \
    --output results/MYDATA/ \
    --weight Training_logs/Unified_R2MoE_MYDATA/models/best_ema.pth \
    --removal_dim 128 --depths 2,2,4,4 \
    --save_mask --save_grid --device cuda:0
```

Multi-GPU (shards the input folder across ranks):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29503 infer_v4.py \
    --output results/MYDATA/ \
    --weight Training_logs/Unified_R2MoE_MYDATA/models/best_ema.pth \
    --save_mask
```

Compute metrics on the outputs:

```bash
pyiqa psnr -t results/MYDATA/removal -r /abs/path/<your_dataset>/test/Out --device cuda --verbose
```

> Trust the **training-time eval PSNR** in `train.log` as the source of truth. A
> standalone infer that rebuilds the model from flags is only correct when every flag
> matches training; `infer_v4.py` warns on missing/unexpected weight keys — heed those
> warnings.

---

## Requirements

See `requirements.txt`. If `mamba_ssm` fails to install, use `torch>=2.6.0+cu124`.
