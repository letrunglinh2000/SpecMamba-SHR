"""
Training script for the unified R2-MoE Specmamba (joint specular detection + removal).

A single Specmamba(dual_head=True) produces:
    mask_logits  -> detection   (CE + Dice)
    D_hat        -> removal      (Charbonnier), via global-residual on the input

Run (4 GPUs):
    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 \
        train_unified.py --opt options/train_unified_SD2.json
    
    CUDA_VISIBLE_DEVICES=0,2,3,4,5 torchrun --nproc_per_node=5 \
    train_unified.py --opt options/train_unified_SHIQ_v4.json
"""
import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import utils_image as util
from utils import utils_option as option
from utils import utils_logger
from utils.utils_dist import get_dist_info, init_dist
from data.select_dataset import define_Dataset
from models.specmamba import Specmamba, moe_balance_loss, regime_prior_loss, reflectance_prompt_loss

import logging


# ----------------------------------------------------------------------------- losses
def charbonnier(x, y, eps=1e-3):
    return torch.sqrt((x - y) ** 2 + eps ** 2).mean()


def fft_l1(pred, target):
    """L1 on the 2D Fourier spectrum (amplitude + phase). Recovers high-freq detail
    that Charbonnier blurs out. Forced to float32 (autocast must not run this in bf16)."""
    with torch.cuda.amp.autocast(enabled=False):
        pf = torch.fft.rfft2(pred.float(), norm="ortho")
        tf = torch.fft.rfft2(target.float(), norm="ortho")
        return (pf - tf).abs().mean()


def _gaussian_window(channels, ksize=11, sigma=1.5, device=None, dtype=None):
    coords = torch.arange(ksize, device=device, dtype=torch.float32) - (ksize - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    w2d = (g[:, None] @ g[None, :])
    w = w2d.expand(channels, 1, ksize, ksize).contiguous()
    return w.to(dtype=dtype) if dtype is not None else w


def ssim_loss(pred, target, ksize=11, sigma=1.5, data_range=1.0):
    """1 - mean SSIM. Forced to float32: the variance terms E[x^2]-E[x]^2 suffer
    catastrophic cancellation in bf16, so autocast must be disabled here."""
    with torch.cuda.amp.autocast(enabled=False):
        pred = pred.float(); target = target.float()
        c = pred.shape[1]
        win = _gaussian_window(c, ksize, sigma, device=pred.device, dtype=pred.dtype)
        pad = ksize // 2
        mu_p = F.conv2d(pred, win, padding=pad, groups=c)
        mu_t = F.conv2d(target, win, padding=pad, groups=c)
        mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
        sig_p = F.conv2d(pred * pred, win, padding=pad, groups=c) - mu_p2
        sig_t = F.conv2d(target * target, win, padding=pad, groups=c) - mu_t2
        sig_pt = F.conv2d(pred * target, win, padding=pad, groups=c) - mu_pt
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        ssim_map = ((2 * mu_pt + c1) * (2 * sig_pt + c2)) / ((mu_p2 + mu_t2 + c1) * (sig_p + sig_t + c2))
        return 1.0 - ssim_map.mean()


def mask_ce_dice(logits, target_long, keep):
    """CE + Dice toward a binary target. keep: (B,) per-sample weight."""
    ce_per = F.cross_entropy(logits, target_long, reduction="none").mean(dim=(1, 2))
    denom = keep.sum().clamp_min(1.0)
    loss_ce = (ce_per * keep).sum() / denom
    probs = F.softmax(logits, dim=1).float()
    oh = F.one_hot(target_long, logits.shape[1]).permute(0, 3, 1, 2).float()
    inter = (probs * oh).sum(dim=(1, 2, 3))
    sets = probs.sum(dim=(1, 2, 3)) + oh.sum(dim=(1, 2, 3))
    dice_per = 1.0 - (2.0 * inter + 1e-6) / (sets + 1e-6)
    loss_dice = (dice_per * keep).sum() / denom
    return loss_ce + loss_dice


# ----------------------------------------------------------------------------- EMA
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


def load_pretrained(bare_model, path):
    if not path or not os.path.isfile(path):
        print(f"[warn] pretrained not found: {path}")
        return
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ("params", "model_state_dict", "state_dict"):
            if isinstance(ckpt.get(k), dict):
                ckpt = ckpt[k]
                break
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
    missing, unexpected = bare_model.load_state_dict(sd, strict=False)
    print(f"[init] warm-start from {os.path.basename(path)}: "
          f"{len(sd) - len(unexpected)} loaded, {len(missing)} missing, {len(unexpected)} unexpected")


# ----------------------------------------------------------------------------- main
def main(json_path="options/train_unified_SD2.json"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default=json_path)
    parser.add_argument("--launcher", default="pytorch")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--dist", action="store_true")
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)
    torchrun_env = any(k in os.environ for k in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))
    opt["dist"] = bool(args.dist or torchrun_env)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if opt["dist"]:
        os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "7200")
        init_dist("pytorch")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    opt["rank"], opt["world_size"] = get_dist_info()
    device = torch.device("cuda")
    is_main = opt["rank"] == 0

    if is_main:
        util.mkdirs((p for k, p in opt["path"].items() if "pretrained" not in k))
        utils_logger.logger_info("train", os.path.join(opt["path"]["log"], "train.log"))
        logger = logging.getLogger("train")

    seed = opt["train"].get("manual_seed") or random.randint(1, 10000)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    tcfg = opt["train"]
    mcfg = opt.get("model_args", {}) or {}
    border = opt.get("scale", 1)

    # ---- data
    for phase, dopt in opt["datasets"].items():
        if phase == "train":
            train_set = define_Dataset(dopt)
            sampler = DistributedSampler(train_set, num_replicas=opt["world_size"], rank=opt["rank"],
                                         shuffle=True, drop_last=True) if opt["dist"] else None
            train_loader = DataLoader(train_set, batch_size=dopt["dataloader_batch_size"],
                                      shuffle=(sampler is None), sampler=sampler,
                                      num_workers=dopt["dataloader_num_workers"], drop_last=True,
                                      pin_memory=True, persistent_workers=dopt["dataloader_num_workers"] > 0)
        elif phase == "test":
            test_set = define_Dataset(dopt)
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2,
                                     drop_last=False, pin_memory=True)

    # ---- model
    net = Specmamba(
        n_channels=3, n_classes=2,
        base_dim=int(mcfg.get("base_dim", 32)),
        depths=tuple(mcfg.get("depths", [2, 2, 4, 4])),
        shared_scan=bool(mcfg.get("shared_scan", True)),
        prompt_dim=int(mcfg.get("prompt_dim", 128)),
        input_range="01",
        stem_stride=int(mcfg.get("stem_stride", 2)),
        lite_expansion=float(mcfg.get("lite_expansion", 1.0)),
        use_mamba_decoder=bool(mcfg.get("use_mamba_decoder", False)),
        use_mamba_stage3=bool(mcfg.get("use_mamba_stage3", True)),
        dual_head=True,
        moe_experts=int(mcfg.get("moe_experts", 4)),
        deep_removal=bool(mcfg.get("deep_removal", False)),
        removal_blocks=int(mcfg.get("removal_blocks", 4)),
        removal_dim=mcfg.get("removal_dim", None),
    ).to(device)

    if opt["path"].get("pretrained"):
        load_pretrained(net, opt["path"]["pretrained"])

    if opt["dist"]:
        net = nn.parallel.DistributedDataParallel(net, device_ids=[torch.cuda.current_device()],
                                                  find_unused_parameters=True)
    bare = net.module if opt["dist"] else net

    ema = EMA(bare, decay=float(tcfg.get("ema_decay", 0.999)))

    opt_g = torch.optim.AdamW(bare.parameters(), lr=float(tcfg["lr"]), weight_decay=0.0)

    # ---- LR schedule: cosine (with linear warmup) or legacy multistep
    sched_type = str(tcfg.get("scheduler", "cosine")).lower()
    _total_iters = int(tcfg.get("total_iters", 100000))
    warmup_iters = int(tcfg.get("warmup_iters", 1000))
    eta_min = float(tcfg.get("eta_min", 1e-6))
    base_lr = float(tcfg["lr"])
    if sched_type == "multistep":
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt_g, tcfg.get("milestones", [20000, 40000, 70000]), gamma=float(tcfg.get("gamma", 0.5)))
    else:
        eta_ratio = eta_min / base_lr

        def _lr_lambda(step):
            # `step` is 1-indexed (sched.step(step) is called after step increments)
            if step < warmup_iters:
                return step / max(1, warmup_iters)
            progress = (step - warmup_iters) / max(1, _total_iters - warmup_iters)
            progress = min(1.0, progress)
            return eta_ratio + (1.0 - eta_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt_g, _lr_lambda)
    if is_main:
        logger.info(f"Scheduler: {sched_type} | base_lr={base_lr:.2e} | warmup={warmup_iters} "
                    f"| total_iters={_total_iters} | eta_min={eta_min:.1e}")

    use_amp = bool(tcfg.get("use_amp", True))
    amp_dtype = torch.bfloat16 if tcfg.get("amp_dtype", "bf16") == "bf16" else torch.float16

    lam_mask = float(tcfg.get("lam_mask", 1.0))
    lam_rem = float(tcfg.get("lam_removal", 1.0))
    lam_bal = float(tcfg.get("lam_bal", 0.01))
    lam_reg = float(tcfg.get("lam_reg", 0.0))
    lam_prompt = float(tcfg.get("lam_prompt", 0.05))
    w_fft = float(tcfg.get("rem_w_fft", 0.1))      # weight of FFT term inside removal loss
    w_ssim = float(tcfg.get("rem_w_ssim", 0.2))    # weight of (1-SSIM) inside removal loss
    total_iters = int(tcfg.get("total_iters", 100000))
    ck_save = int(tcfg.get("checkpoint_save", 2500))
    ck_test = int(tcfg.get("checkpoint_test", 2500))
    ck_print = int(tcfg.get("checkpoint_print", 100))
    eval_max = int(tcfg.get("eval_max", 100))
    save_images = bool(tcfg.get("save_images", False))

    if is_main:
        n_params = sum(p.numel() for p in bare.parameters() if p.requires_grad)
        logger.info(f"Unified R2-MoE Specmamba | params={n_params/1e6:.2f}M | seed={seed}")

    best_psnr, best_step = 0.0, 0
    step = 0
    net.train()
    epoch = 0
    while step < total_iters:
        if opt["dist"]:
            sampler.set_epoch(epoch)
        for data in train_loader:
            step += 1
            if step > total_iters:
                break
            sched.step(step)

            L = data["L"].to(device, non_blocking=True)            # (B,3,H,W) input
            H = data["H"].to(device, non_blocking=True)            # (B,4,H,W) quaternion GT
            M = data["M"].to(device, non_blocking=True)            # (B,C,H,W) mask
            has_mask = data.get("has_mask", torch.ones(L.shape[0])).to(device).float().view(-1)
            H_rgb = H[:, 1:]                                       # (B,3,H,W) clean diffuse

            opt_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                mask_logits, D_hat, aux = net(L, return_aux=True)
                gate = aux["regime_gate"]

                target = (M[:, 0] >= 0.5).long()                  # (B,H,W)
                loss_mask = mask_ce_dice(mask_logits, target, has_mask)
                # Removal = Charbonnier + FFT(high-freq) + (1-SSIM)(structure).
                # Charbonnier alone plateaus (blur-biased); FFT/SSIM target the removal ceiling.
                loss_char = charbonnier(D_hat, H_rgb)
                loss_fft = fft_l1(D_hat, H_rgb) if w_fft > 0 else D_hat.new_zeros(())
                loss_ssim = ssim_loss(D_hat, H_rgb) if w_ssim > 0 else D_hat.new_zeros(())
                loss_rem = loss_char + w_fft * loss_fft + w_ssim * loss_ssim
                loss_bal = moe_balance_loss(gate)
                if lam_reg > 0:
                    spec = (M[:, :1] >= 0.5).float()
                    sat = (L.amax(dim=1, keepdim=True) > 0.98).float()
                    spec_s = F.interpolate(spec, size=gate.shape[-2:], mode="nearest")
                    sat_s = F.interpolate(sat, size=gate.shape[-2:], mode="nearest")
                    loss_reg = regime_prior_loss(gate, spec_s, sat_s)
                else:
                    loss_reg = D_hat.new_zeros(())

                # Reflectance-prompt weak supervision on the prompt bank (paper contribution):
                # diversity + routing loss over prompt_weight [B,6] given the visual prior [B,4,H,W].
                if lam_prompt > 0:
                    loss_prompt, _ = reflectance_prompt_loss(aux["prompt_weight"].float(),
                                                             aux["visual_prior"].float())
                else:
                    loss_prompt = D_hat.new_zeros(())

                total = (lam_mask * loss_mask + lam_rem * loss_rem + lam_bal * loss_bal
                         + lam_reg * loss_reg + lam_prompt * loss_prompt)

            total.backward()
            opt_g.step()
            ema.update(bare)

            if is_main and step % ck_print == 0:
                logger.info(f"<iter:{step:7,d}, lr:{opt_g.param_groups[0]['lr']:.2e}> "
                            f"loss:{total.item():.4e} L_mask:{loss_mask.item():.4e} "
                            f"L_rem:{loss_rem.item():.4e} (char:{float(loss_char):.4e} "
                            f"fft:{float(loss_fft):.4e} ssim:{float(loss_ssim):.4e}) "
                            f"L_bal:{loss_bal.item():.4e} L_reg:{float(loss_reg):.4e} L_prompt:{float(loss_prompt):.4e}")

            if step % ck_save == 0 and is_main:
                torch.save(bare.state_dict(), os.path.join(opt["path"]["models"], f"{step}_net.pth"))
                torch.save(ema.shadow, os.path.join(opt["path"]["models"], f"{step}_ema.pth"))
                logger.info("Saved model.")

            # ---- eval (rank-0, EMA), DDP-safe with barriers
            if step % ck_test == 0 and opt["dist"]:
                torch.distributed.barrier()
            if step % ck_test == 0 and is_main:
                eval_net = Specmamba(
                    n_channels=3, n_classes=2, base_dim=int(mcfg.get("base_dim", 32)),
                    depths=tuple(mcfg.get("depths", [2, 2, 4, 4])), shared_scan=bool(mcfg.get("shared_scan", True)),
                    prompt_dim=int(mcfg.get("prompt_dim", 128)), input_range="01",
                    stem_stride=int(mcfg.get("stem_stride", 2)), lite_expansion=float(mcfg.get("lite_expansion", 1.0)),
                    use_mamba_decoder=bool(mcfg.get("use_mamba_decoder", False)),
                    use_mamba_stage3=bool(mcfg.get("use_mamba_stage3", True)),
                    dual_head=True, moe_experts=int(mcfg.get("moe_experts", 4)),
                    deep_removal=bool(mcfg.get("deep_removal", False)),
                    removal_blocks=int(mcfg.get("removal_blocks", 4)),
                    removal_dim=mcfg.get("removal_dim", None),
                ).to(device)
                eval_net.load_state_dict(ema.shadow, strict=True)
                eval_net.eval()
                torch.cuda.empty_cache()
                psnr_sum = ssim_sum = 0.0
                idx = 0
                with torch.no_grad():
                    for td in test_loader:
                        if idx >= eval_max:
                            break
                        idx += 1
                        Lt = td["L"].to(device)
                        Ht = td["H"].to(device)[:, 1:]
                        _, D_t = eval_net(Lt, return_aux=False)
                        E_img = util.tensor2uint(D_t[0].float().cpu())
                        H_img = util.tensor2uint(Ht[0].float().cpu())
                        psnr_sum += util.calculate_psnr(E_img, H_img, border=border, test_y_channel=False)
                        ssim_sum += util.calculate_ssim(E_img, H_img, border=0, scale=1)
                        if save_images:
                            nm = os.path.splitext(os.path.basename(td["L_path"][0]))[0]
                            util.imsave(E_img, os.path.join(opt["path"]["images"], f"{nm}_{step}.png"))
                del eval_net
                if idx > 0:
                    avg_p, avg_s = psnr_sum / idx, ssim_sum / idx
                    is_best = avg_p > best_psnr
                    if is_best:
                        best_psnr, best_step = avg_p, step
                        torch.save(ema.shadow, os.path.join(opt["path"]["models"], "best_ema.pth"))
                    logger.info(f" Eval (n={idx}): PSNR={avg_p:.2f}dB SSIM={avg_s:.4f}"
                                f"  {'<-- best' if is_best else f'(best={best_psnr:.2f}@{best_step:,})'}\n")
            if step % ck_test == 0 and opt["dist"]:
                torch.distributed.barrier()
        epoch += 1


if __name__ == "__main__":
    main()
