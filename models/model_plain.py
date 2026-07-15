"""
Diffusion training for SpecMamba + Qformer (x_0-prediction).

Forward (corruption) process is physics-grounded:
    h    = L - H_rgb                          (specular residual)
    x_t  = H_rgb + alpha_t * h                (interpolated along H -> L)
    x_T  = L,  x_0 = H_rgb

Both networks predict at time t:
    M_t   = SpecMamba(L_512, t_emb)           (time-conditioned mask)
    H_hat = Qformer(cat(M_t, x_t), t_emb)     (x_0-prediction, 4-ch quaternion)
    h_hat = (x_t - H_hat_rgb) / alpha_t       (analytical residual)

Losses (single backward, x_0 parameterisation):
    L_recon    = ||H_hat_rgb - H_rgb||_1
    L_mask     = CE+Dice(M_t, M_gt)           (only when α_t > τ)
    L_score    = alpha_t * ||h_hat - h||_2^2  (mild, x_0-stable weighting)
    L_cycle    = ||(H_hat_rgb + M_t * max(L-H_hat_rgb,0)) - L||_1
    L_sparse   = alpha_t * mean(M_t)

Speed wins vs the previous version:
    - Single forward + single backward (was 2x both).
    - AMP autocast + GradScaler.
    - cudnn.benchmark and TF32 (set by the training script).
    - SpecMamba now DDP-wrapped (was running independently per GPU).
    - find_unused_parameters=False (all params hit by the gradient).
    - Vectorised EMA via torch._foreach_*.
"""

from collections import OrderedDict
from typing import Dict, Optional
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam, lr_scheduler

from models.diffusion_modules import (
    AlphaSchedule,
    DiffusionAnyIR,
    DiffusionQformer,
    DiffusionSpecMamba,
    TimeEmbedMLP,
    load_pretrained_into_wrapper,
)
from models.loss import CharbonnierLoss
from models.loss_ssim import SSIMLoss
from models.model_base import ModelBase
from models.select_network import define_G
from models.specmamba import Specmamba, reflectance_prompt_loss


# ---------------------------------------------------------------------------
# Dice helpers (kept from the previous version)
# ---------------------------------------------------------------------------

def dice_coeff(inp: Tensor, target: Tensor, reduce_batch_first: bool = False, eps: float = 1e-6):
    assert inp.size() == target.size()
    if inp.dim() == 2 or reduce_batch_first:
        inter = torch.dot(inp.reshape(-1), target.reshape(-1))
        sets_sum = inp.sum() + target.sum()
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        return (2 * inter + eps) / (sets_sum + eps)
    dice = 0.0
    for i in range(inp.shape[0]):
        dice = dice + dice_coeff(inp[i], target[i])
    return dice / inp.shape[0]


def multiclass_dice(inp: Tensor, target: Tensor, eps: float = 1e-6):
    assert inp.size() == target.size()
    d = 0.0
    for c in range(inp.shape[1]):
        d = d + dice_coeff(inp[:, c], target[:, c], reduce_batch_first=True, eps=eps)
    return d / inp.shape[1]


def dice_loss(inp: Tensor, target: Tensor, multiclass: bool = True):
    fn = multiclass_dice if multiclass else dice_coeff
    return 1 - fn(inp, target)


# ---------------------------------------------------------------------------
# ModelPlain
# ---------------------------------------------------------------------------

class ModelPlain(ModelBase):
    """Diffusion training (x_0-prediction) with time-conditioned SpecMamba + Qformer."""

    def __init__(self, opt):
        super().__init__(opt)
        self.opt_train = self.opt["train"]

        # Diffusion config (with sane defaults).
        dcfg = self.opt_train.get("diffusion", {}) or {}
        self.T = int(dcfg.get("T", 8))
        self.t_dim = int(dcfg.get("t_dim", 128))
        self.alpha_sched = AlphaSchedule(self.T).to(self.device)
        # Curriculum: lerp t-min from T down to `t_min_floor` over `t_warmup` steps.
        self.t_warmup = int(dcfg.get("t_warmup", 5000))
        # Lowest t the curriculum samples. Default 1 = full diffusion range.
        # Set >1 (e.g. 5) to keep training in the high-t regime that
        # infer_steps=1 inference actually uses — prevents the single-step
        # quality from being diluted by easy low-t (near-clean) cases.
        self.t_min_floor = int(dcfg.get("t_min_floor", 1))
        # Loss weights.
        self.lam_recon = float(dcfg.get("lam_recon", 1.0))
        self.lam_mask = float(dcfg.get("lam_mask", 1.0))
        self.lam_score = float(dcfg.get("lam_score", 0.5))
        self.lam_cycle = float(dcfg.get("lam_cycle", 0.5))
        self.lam_sparse = float(dcfg.get("lam_sparse", 0.01))
        # Mask supervision is only meaningful where the highlight is still present.
        self.mask_alpha_thresh = float(dcfg.get("mask_alpha_thresh", 0.3))
        # Inference: number of DDIM steps.
        self.infer_steps = int(dcfg.get("infer_steps", self.T))
        # SpecMamba input resolution. None means use the current crop/input size.
        self.specmamba_input_size = dcfg.get("specmamba_input_size", None)
        # SpecMamba freeze schedule. During the warmup window, SpecMamba is
        # used as a fixed mask prior; after that it is unfrozen at a small LR.
        self.specmamba_freeze_warmup = int(dcfg.get("specmamba_freeze_warmup", 0))
        self.specmamba_unfreeze_lr_mult = float(dcfg.get("specmamba_unfreeze_lr_mult", 0.2))
        self._specmamba_is_frozen = False
        self._specmamba_unfreeze_lr_scaled = False

        # ---- Architecture / training upgrades (Tier 1-2 from review) -----
        # Multi-scale mask injection inside the restoration backbone.
        self.use_mask_film = bool(dcfg.get("use_mask_film", True))
        # Predict the specular layer S and compute H_hat = (L - S).clamp(0,1).
        self.predict_residual = bool(dcfg.get("predict_residual", True))
        # Self-conditioning: train on the model's own H_hat with this probability.
        self.self_cond_prob = float(dcfg.get("self_cond_prob", 0.5))
        # alpha-weighted L_recon: spends gradient on hard (high-alpha) cases.
        self.alpha_weight_recon = bool(dcfg.get("alpha_weight_recon", True))
        # Drop L_score: it is mathematically redundant with L_recon under the
        # x_0 / residual parameterisation. lam_score in the JSON is ignored
        # when this flag is on.
        self.drop_score_loss = bool(dcfg.get("drop_score_loss", True))
        # LPIPS perceptual loss weight (0 disables).
        self.lam_perc = float(dcfg.get("lam_perc", 0.0))
        # SSIM loss weight (complement to L1 for color/structure fidelity).
        self.lam_ssim = float(dcfg.get("lam_ssim", 0.0))
        # Mask-weighted reconstruction: upweight L1 inside specular regions.
        self.lam_spec_recon = float(dcfg.get("lam_spec_recon", 0.0))

        # ---- PCC: Physically-Controlled Consistency (single-stage paradigm) ----
        # paradigm = "diffusion" (default, existing) or "pcc".
        self.paradigm = str(self.opt_train.get("paradigm", "diffusion")).lower()
        pcc = self.opt_train.get("pcc", {}) or {}
        self.pcc_n = int(pcc.get("n", 4))                       # family size (1 real + n-1 synthetic)
        self.pcc_lam_sup = float(pcc.get("lam_sup", 1.0))       # supervised vs D_gt
        self.pcc_lam_cons = float(pcc.get("lam_cons", 0.5))     # specular-invariance consistency
        self.pcc_lam_sat = float(pcc.get("lam_sat", 0.5))       # saturated-core weighting
        self.pcc_lam_mask = float(pcc.get("lam_mask", 0.2))     # SpecMamba supervision
        self.pcc_blobs = tuple(pcc.get("blobs", [2, 4]))        # min/max blobs per synthetic highlight
        self.pcc_beta_range = tuple(pcc.get("beta_range", [0.2, 0.8]))  # highlight magnitude range
        self.pcc_w_real = float(pcc.get("w_real", 1.0))         # weight on real-sample reconstruction
        self.pcc_w_syn = float(pcc.get("w_syn", 0.3))           # weight on synthetic-sample reconstruction
        self.pcc_lam_prompt = float(pcc.get("lam_prompt", 0.0)) # reflectance-prompt weak supervision (0 = off)

        # AMP / precision.
        self.use_amp = bool(self.opt_train.get("use_amp", True))
        amp_dtype = self.opt_train.get("amp_dtype", "fp16")
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

        # ----- build networks -----
        # Backbone choice: "qformer" (default, quaternion U-Net) or "anyir"
        # (Restormer-style transformer with degradation-adaption blocks).
        self.backbone = str(self.opt_train.get("backbone", "qformer")).lower()
        if self.backbone == "anyir":
            from models.network_anyir import AnyIR
            anyir_cfg = self.opt_train.get("anyir", {}) or {}

            def _build_anyir():
                return AnyIR(
                    inp_channels=4,        # cat(mask, RGB) — matches Qformer interface
                    out_channels=4,        # first channel ignored downstream (parity with quaternion)
                    dim=int(anyir_cfg.get("dim", 28)),
                    num_blocks=list(anyir_cfg.get("num_blocks", [3, 5, 5, 7])),
                    num_refinement_blocks=int(anyir_cfg.get("num_refinement_blocks", 4)),
                    heads=list(anyir_cfg.get("heads", [1, 2, 4, 8])),
                    ffn_expansion_factor=float(anyir_cfg.get("ffn_expansion_factor", 2)),
                )

            wrap_kwargs = dict(
                t_dim=self.t_dim,
                use_mask_film=self.use_mask_film,
                predict_residual=self.predict_residual,
            )
            base_q = _build_anyir()
            self.netG = DiffusionAnyIR(base_q, **wrap_kwargs).to(self.device)
            self.netG = self.model_to_device(self.netG)

            if self.opt_train["E_decay"] > 0:
                base_qE = _build_anyir()
                self.netE = DiffusionAnyIR(base_qE, **wrap_kwargs).to(self.device).eval()
                for p in self.netE.parameters():
                    p.requires_grad = False
        else:
            wrap_kwargs = dict(
                t_dim=self.t_dim,
                use_mask_film=self.use_mask_film,
                predict_residual=self.predict_residual,
            )
            base_q = define_G(opt)
            self.netG = DiffusionQformer(base_q, **wrap_kwargs).to(self.device)
            self.netG = self.model_to_device(self.netG)

            if self.opt_train["E_decay"] > 0:
                base_qE = define_G(opt)
                self.netE = DiffusionQformer(base_qE, **wrap_kwargs).to(self.device).eval()
                for p in self.netE.parameters():
                    p.requires_grad = False

        spec_kwargs = dict(
            base_dim=32,
            depths=(2, 2, 4, 4),
            shared_scan=True,
            prompt_dim=128,
            input_range="01",
            stem_stride=2,
            lite_expansion=1.0,
            use_mamba_decoder=False,
            use_mamba_stage3=True,
        )
        base_spec = Specmamba(**spec_kwargs)
        self.SpecMamba = DiffusionSpecMamba(base_spec, t_dim=self.t_dim).to(self.device)
        self.SpecMamba = self.model_to_device(self.SpecMamba)

        # Shared time embedding MLP — DDP-wrapped so gradients are all-reduced.
        self.t_embed = TimeEmbedMLP(dim=self.t_dim).to(self.device)
        self.t_embed = self.model_to_device(self.t_embed)

        # GradScaler for AMP (no-op when use_amp=False or bf16).
        self._use_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._use_scaler)

        # Gradient clipping (max global norm). null/None disables.
        clipv = self.opt_train.get("G_optimizer_clipgrad")
        self.clip_grad_norm = float(clipv) if clipv is not None else None
        # Count of non-finite steps skipped (logged for monitoring).
        self._nan_skip_count = 0

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def init_train(self, resume_step: int = 0):
        self.load()
        self.netG.train()
        self.SpecMamba.train()
        self.t_embed.train()
        self.define_loss()
        self.define_optimizer()
        self.load_optimizers()
        self.define_scheduler(resume_step)
        self.log_dict: Dict[str, float] = OrderedDict()

    def init_test(self):
        self.load()
        self.netG.eval()
        self.SpecMamba.eval()
        self.t_embed.eval()

    # -----------------------------------------------------------------
    # Pretrained loading (handles "params" wrapping and "module." prefix)
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_state(ckpt):
        if not isinstance(ckpt, dict):
            return ckpt
        for k in ("params", "model_state_dict", "state_dict"):
            v = ckpt.get(k)
            if isinstance(v, dict):
                return v
        return ckpt

    @staticmethod
    def _strip_module(state_dict):
        out = OrderedDict()
        for k, v in state_dict.items():
            out[k[7:] if k.startswith("module.") else k] = v
        return out

    def _load_one(self, load_path: str, wrapper_net: nn.Module, label: str):
        if load_path is None:
            return
        print(f"Loading pretrained {label} from [{load_path}] ...")
        ckpt = torch.load(load_path, map_location=self.device)
        state_dict = self._strip_module(self._extract_state(ckpt))
        bare = self.get_bare_model(wrapper_net)
        load_pretrained_into_wrapper(bare, state_dict)

    def load(self):
        self._load_one(self.opt["path"]["pretrained_netG"], self.netG, "Qformer")
        if self.opt_train["E_decay"] > 0:
            path_E = self.opt["path"]["pretrained_netE"]
            if path_E is not None:
                self._load_one(path_E, self.netE, "Qformer-EMA")
            else:
                print("Copying netG weights into netE ...")
                self._copy_G_into_E()
            self.netE.eval()
        self._load_one(self.opt["path"]["pretrained_netSpecMamba"], self.SpecMamba, "SpecMamba")
        # Optional: shared time-embedding MLP checkpoint (only after a resume).
        path_t = self.opt["path"].get("pretrained_t_embed")
        if path_t is not None:
            print(f"Loading t_embed [{path_t}] ...")
            self.get_bare_model(self.t_embed).load_state_dict(
                torch.load(path_t, map_location=self.device)
            )

    def load_optimizers(self):
        if self.opt_train.get("G_optimizer_reuse"):
            p = self.opt["path"]["pretrained_optimizerG"]
            if p is not None:
                print(f"Loading optimizerG [{p}] ...")
                self.load_optimizer(p, self.G_optimizer)
            p = self.opt["path"]["pretrained_optimizerSpecMamba"]
            if p is not None:
                print(f"Loading optimizerSpecMamba [{p}] ...")
                self.load_optimizer(p, self.SpecMamba_optimizer)

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, "G", iter_label)
        self.save_network(self.save_dir, self.SpecMamba, "SpecMamba", iter_label)
        if self.opt_train["E_decay"] > 0:
            self.save_network(self.save_dir, self.netE, "E", iter_label)
        # Save the shared time-embedding MLP (strip DDP wrapping).
        import os
        t_path = os.path.join(self.save_dir, f"{iter_label}_t_embed.pth")
        bare_t = self.get_bare_model(self.t_embed)
        torch.save({k: v.cpu() for k, v in bare_t.state_dict().items()}, t_path)
        if self.opt_train.get("G_optimizer_reuse"):
            self.save_optimizer(self.save_dir, self.G_optimizer, "optimizerG", iter_label)
            self.save_optimizer(self.save_dir, self.SpecMamba_optimizer, "optimizerSpecMamba", iter_label)

    # -----------------------------------------------------------------
    # Loss / optimizer / scheduler
    # -----------------------------------------------------------------

    def define_loss(self):
        kind = self.opt_train["G_lossfn_type"]
        if kind == "l1":
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif kind == "l2":
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif kind == "l2sum":
            self.G_lossfn = nn.MSELoss(reduction="sum").to(self.device)
        elif kind == "ssim":
            self.G_lossfn = SSIMLoss().to(self.device)
        elif kind == "charbonnier":
            self.G_lossfn = CharbonnierLoss(self.opt_train["G_charbonnier_eps"]).to(self.device)
        else:
            raise NotImplementedError(kind)
        self.G_lossfn_weight = self.opt_train["G_lossfn_weight"]
        self.criterion = nn.CrossEntropyLoss()
        if self.lam_ssim > 0:
            self.ssim_criterion = SSIMLoss().to(self.device)

    def define_optimizer(self):
        # Two LR groups: pretrained backbone (slow) vs newly-added FiLM/time modules (fast).
        lr = self.opt_train["G_optimizer_lr"]
        lr_new = float(self.opt_train.get("new_module_lr_mult", 10.0)) * lr

        def split(module):
            slow, fast = [], []
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                if any(tok in name for tok in ("film_", "time_to_prompt", "t_embed")):
                    fast.append(p)
                else:
                    slow.append(p)
            return slow, fast

        g_slow, g_fast = split(self.netG)
        s_slow, s_fast = split(self.SpecMamba)
        t_params = list(self.t_embed.parameters())

        self.G_optimizer = Adam(
            [{"params": g_slow, "lr": lr},
             {"params": g_fast + t_params, "lr": lr_new}],
            weight_decay=0,
        )
        spec_lr = lr / 5.0
        self.specmamba_base_lr = spec_lr
        self.specmamba_fast_lr = spec_lr * 10.0
        self.SpecMamba_optimizer = Adam(
            [{"params": s_slow, "lr": spec_lr},
             {"params": s_fast, "lr": spec_lr * 10.0}],
            weight_decay=0,
        )
        if self.specmamba_freeze_warmup > 0:
            self._set_specmamba_frozen(True)

    def define_scheduler(self, resume_step: int = 0):
        ms = self.opt_train["G_scheduler_milestones"]
        gm = self.opt_train["G_scheduler_gamma"]
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer, ms, gm, last_epoch=resume_step - 1 if resume_step > 0 else -1))
        self.schedulers.append(lr_scheduler.MultiStepLR(self.SpecMamba_optimizer, ms, gm, last_epoch=resume_step - 1 if resume_step > 0 else -1))

    def _set_specmamba_frozen(self, frozen: bool):
        """Freeze/unfreeze SpecMamba while keeping it inside its optimizer."""
        if self._specmamba_is_frozen == frozen:
            return
        bare_spec = self.get_bare_model(self.SpecMamba)
        for p in bare_spec.parameters():
            p.requires_grad = not frozen
        self._specmamba_is_frozen = frozen

    def _scale_specmamba_lr_after_scheduler(self):
        # Scale base_lrs (not the live group["lr"]): MultiStepLR recomputes
        # group["lr"] from base_lrs * gamma^k on every scheduler.step(), so a
        # direct edit to group["lr"] would be clobbered on the next iteration.
        mult = self.specmamba_unfreeze_lr_mult
        spec_scheduler = self.schedulers[1]  # [0]=G, [1]=SpecMamba (define_scheduler order)
        spec_scheduler.base_lrs = [b * mult for b in spec_scheduler.base_lrs]
        for group in self.SpecMamba_optimizer.param_groups:
            group["lr"] *= mult

    def _update_specmamba_freeze_schedule(self, current_step: int):
        if self.specmamba_freeze_warmup <= 0:
            return
        should_freeze = current_step <= self.specmamba_freeze_warmup
        self._set_specmamba_frozen(should_freeze)
        if not should_freeze and not self._specmamba_unfreeze_lr_scaled:
            self._scale_specmamba_lr_after_scheduler()
            self._specmamba_unfreeze_lr_scaled = True

    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------

    def _specmamba_size(self, spatial_size):
        if self.specmamba_input_size is None:
            return spatial_size
        if isinstance(self.specmamba_input_size, int):
            return (self.specmamba_input_size, self.specmamba_input_size)
        if isinstance(self.specmamba_input_size, (list, tuple)) and len(self.specmamba_input_size) == 2:
            return tuple(int(v) for v in self.specmamba_input_size)
        raise ValueError(f"Invalid specmamba_input_size: {self.specmamba_input_size}")

    def feed_data(self, data, need_H: bool = True):
        del need_H  # API compat; H is always loaded for diffusion training
        self.L = data["L"].to(self.device, non_blocking=True)
        self.M = data["M"].to(self.device, non_blocking=True)
        self.H = data["H"].to(self.device, non_blocking=True)
        self.has_mask = data.get("has_mask", torch.ones(self.L.shape[0])).to(
            self.device, non_blocking=True
        ).float().view(-1)
        # GT mask at the configured SpecMamba supervision size (binary).
        spec_size = self._specmamba_size(self.L.shape[-2:])
        self.M_spec = F.interpolate(self.M, size=spec_size, mode="nearest")[:, 0]
        self.M_spec = (self.M_spec >= 10 / 255.0).long()

    # -----------------------------------------------------------------
    # Curriculum on t
    # -----------------------------------------------------------------

    def _sample_t(self, batch_size: int, step: int) -> Tensor:
        """Curriculum: start near T, expand down to t_min_floor over t_warmup steps."""
        floor = self.t_min_floor
        if self.t_warmup <= 0:
            t_lo = floor
        else:
            frac = min(1.0, step / max(1, self.t_warmup))
            t_lo = max(floor, int(round(self.T - frac * (self.T - floor))))
        # uniform on [t_lo, T]
        return torch.randint(t_lo, self.T + 1, (batch_size,), device=self.device)

    # -----------------------------------------------------------------
    # Forward (training)
    # -----------------------------------------------------------------

    def _alpha_view(self, t: Tensor) -> Tensor:
        return self.alpha_sched.alpha(t).view(-1, 1, 1, 1)

    def _forward_train(self, t: Tensor):
        """Single time-conditioned forward.

        SpecMamba sees the corrupted image x_t (resized to 512). At t=T this
        equals L (SpecMamba's pretraining input); for t<T the mask refines
        along the denoising trajectory.
        """
        H_rgb = self.H[:, 1:]                                # (B,3,H,W) clean
        h_gt = self.L - H_rgb                                # (B,3,H,W) residual
        alpha = self._alpha_view(t)                          # (B,1,1,1)
        x_t = H_rgb + alpha * h_gt                           # (B,3,H,W)

        # Time embedding
        t_emb = self.t_embed(t)                              # (B, t_dim)

        # SpecMamba on x_t at the configured supervision size. By default this
        # matches the current crop size, avoiding an expensive forced 512x512 pass.
        # time-agnostic: it segments specular content from whatever image it
        # receives, regardless of how denoised that image is.
        spec_size = self._specmamba_size(H_rgb.shape[-2:])
        x_t_spec = F.interpolate(x_t, size=spec_size, mode="bilinear", align_corners=False)
        if self._specmamba_is_frozen:
            with torch.no_grad():
                mask_logits_spec = self.SpecMamba(x_t_spec)      # (B,2,h_spec,w_spec)
            mask_logits_spec = mask_logits_spec.detach()
        else:
            mask_logits_spec = self.SpecMamba(x_t_spec)          # (B,2,h_spec,w_spec)
        mask_logits = F.interpolate(
            mask_logits_spec, size=H_rgb.shape[-2:], mode="bilinear", align_corners=False
        )
        spec_prob = mask_logits.softmax(dim=1)[:, 1:2]       # (B,1,H,W) specular probability

        # Detach the mask before the reconstruction backbone. SpecMamba is then
        # supervised ONLY by the mask loss (CE+Dice toward GT). Without this,
        # L_recon (5x larger than L_mask) flows back into the mask and drags it
        # toward recon-convenient blobs — corrupting irregular masks over training
        # while leaving simple/circular ones intact. The reconstruction still uses
        # the mask as a (correct) conditioning input.
        spec_prob_cond = spec_prob.detach()

        # Qformer input: cat(M_t, x_t) — channel 0 = mask, channels 1..3 = x_t (matches pretraining)
        q_in = torch.cat([spec_prob_cond, x_t], dim=1)       # (B,4,H,W)
        q_out = self.netG(q_in, t_emb, mask=spec_prob_cond)  # (B,4,H,W) quaternion
        if self.predict_residual:
            # q_out is predicted specular layer S; recover clean image
            H_hat_rgb = (x_t - q_out[:, 1:]).clamp(0, 1)    # (B,3,H,W)
        else:
            H_hat_rgb = q_out[:, 1:]                         # (B,3,H,W) x_0-prediction

        # Analytical residual prediction (used by score loss & inference DDIM step).
        # Clamp alpha to 0.05 min so h_hat stays finite in fp16: at t=0 alpha≈1.5e-4,
        # diff/1.5e-4 easily exceeds fp16 max (65504), making h_hat²=inf → NaN loss.
        h_hat = (x_t - H_hat_rgb) / alpha.clamp_min(0.05)

        return {
            "alpha": alpha,
            "x_t": x_t,
            "h_gt": h_gt,
            "H_rgb": H_rgb,
            "H_hat_rgb": H_hat_rgb,
            "h_hat": h_hat,
            "spec_prob": spec_prob,
            "mask_logits_spec": mask_logits_spec,
        }

    def _compute_loss(self, fwd: Dict[str, Tensor]):
        H_rgb = fwd["H_rgb"]
        H_hat_rgb = fwd["H_hat_rgb"]
        h_gt = fwd["h_gt"]
        h_hat = fwd["h_hat"]
        spec_prob = fwd["spec_prob"]
        alpha = fwd["alpha"]
        mask_logits_spec = fwd["mask_logits_spec"]

        # ---- reconstruction (x_0-prediction) ----
        # Mask-upweighted L1: specular pixels get (1 + lam_spec_recon) × weight so
        # the network pays proportionally more attention to restoring highlight colors.
        if self.lam_spec_recon > 0:
            recon_w = 1.0 + self.lam_spec_recon * spec_prob.detach()  # (B,1,H,W)
            loss_recon = (recon_w * (H_hat_rgb - H_rgb).abs()).mean()
        else:
            loss_recon = F.l1_loss(H_hat_rgb, H_rgb)

        # ---- residual score-matching (alpha-weighted to cancel 1/alpha blow-up)
        # Gated by drop_score_loss (default True) since it is mathematically
        # redundant with L_recon under x_0-parameterisation.
        if not self.drop_score_loss:
            loss_score = (alpha * (h_hat - h_gt).pow(2)).mean()
        else:
            loss_score = H_hat_rgb.new_zeros(1).squeeze()

        # ---- mask supervision, gated PER-SAMPLE by alpha_t > thresh ------------
        # Two branches, graph-identical across DDP (no boolean indexing):
        #   has_mask=1 → GT binary CE + Dice at SpecMamba resolution
        #   has_mask=0 → soft pseudo-mask from h_gt = L−H, supervised with BCE
        alpha_gate = (alpha.view(-1) > self.mask_alpha_thresh).float()               # (B,)

        # --- branch A: GT binary mask (has_mask=1) ---
        keep = alpha_gate * self.has_mask                                            # (B,)
        ce_per = F.cross_entropy(mask_logits_spec, self.M_spec, reduction='none')    # (B, H, W)
        ce_per = ce_per.mean(dim=(1, 2))                                             # (B,)
        denom = keep.sum().clamp_min(1.0)
        loss_ce = (ce_per * keep).sum() / denom

        probs = F.softmax(mask_logits_spec, dim=1).float()                           # (B, 2, H, W)
        target_oh = F.one_hot(self.M_spec, 2).permute(0, 3, 1, 2).float()            # (B, 2, H, W)
        inter = (probs * target_oh).sum(dim=(1, 2, 3))                               # (B,)
        sets = probs.sum(dim=(1, 2, 3)) + target_oh.sum(dim=(1, 2, 3))               # (B,)
        dice_per = 1.0 - (2.0 * inter + 1e-6) / (sets + 1e-6)                        # (B,)
        loss_dice = (dice_per * keep).sum() / denom

        # --- branch B: no GT mask (has_mask=0) — soft pseudo-mask from pair diff ---
        # h_gt = L − H is bright at highlight regions and near-zero elsewhere, so
        # its per-channel max, normalized per image, is a valid soft proxy for the
        # specular mask. Stops gradient so SpecMamba trains toward a stable target.
        with torch.no_grad():
            h_soft = h_gt.clamp(min=0).max(dim=1, keepdim=True)[0]                  # (B,1,H,W)
            spec_h, spec_w = mask_logits_spec.shape[-2:]
            h_soft = F.interpolate(h_soft, size=(spec_h, spec_w),
                                   mode='bilinear', align_corners=False)
            h_max = h_soft.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            h_soft = (h_soft / h_max).clamp(0.0, 1.0)                               # (B,1,h,w) in [0,1]

        # sigmoid(logit_1 - logit_0) == softmax_1 for 2-class; BCEWithLogits is AMP-safe
        soft_logit = mask_logits_spec[:, 1:2] - mask_logits_spec[:, 0:1]            # (B,1,h,w)
        soft_bce_per = F.binary_cross_entropy_with_logits(
            soft_logit.float(), h_soft.float(), reduction='none'
        ).mean(dim=(1, 2, 3))                                                        # (B,)
        no_mask = 1.0 - self.has_mask                                                # (B,)
        soft_keep = alpha_gate * no_mask                                             # (B,)
        soft_denom = soft_keep.sum().clamp_min(1.0)
        loss_soft = (soft_bce_per * soft_keep).sum() / soft_denom

        loss_mask = loss_ce + loss_dice + loss_soft

        # ---- physics cycle: L should be recovered as Hhat + M * spec_residual ----
        s_hat = (self.L - H_hat_rgb).clamp(min=0)
        L_recon = H_hat_rgb + spec_prob * s_hat
        loss_cycle = F.l1_loss(L_recon, self.L)

        # ---- sparsity prior (mask should be small, scaled by alpha) ----
        loss_sparse = (alpha.view(-1, 1, 1, 1) * spec_prob).mean()

        # ---- SSIM loss (color/structure fidelity complement to L1) ----
        if self.lam_ssim > 0:
            loss_ssim = 1.0 - self.ssim_criterion(H_hat_rgb.float(), H_rgb.float())
        else:
            loss_ssim = H_hat_rgb.new_zeros(1).squeeze()

        total = (
            self.lam_recon * loss_recon
            + self.lam_score * loss_score
            + self.lam_mask * loss_mask
            + self.lam_cycle * loss_cycle
            + self.lam_sparse * loss_sparse
            + self.lam_ssim * loss_ssim
        )
        logs = {
            "loss": total.detach(),
            "L_recon": loss_recon.detach(),
            "L_score": loss_score.detach(),
            "L_mask": loss_mask.detach() if torch.is_tensor(loss_mask) else loss_mask,
            "L_soft": loss_soft.detach(),
            "L_cycle": loss_cycle.detach(),
            "L_sparse": loss_sparse.detach(),
            "L_ssim": loss_ssim.detach(),
        }
        return total, logs

    # -----------------------------------------------------------------
    # One training step
    # -----------------------------------------------------------------

    def _clip_grads(self) -> bool:
        """Clip global grad norm across all trainable params (call after unscale).

        Returns False (and zeroes grads) if any grad is non-finite, so the caller
        can skip the optimizer step. This avoids the clip_grad_norm_ 0*inf trap:
        with an inf grad, total_norm=inf -> clip_coef=0 -> 0*inf=NaN poisons weights.
        """
        params = [p for p in self.netG.parameters() if p.grad is not None]
        params += [p for p in self.t_embed.parameters() if p.grad is not None]
        if not self._specmamba_is_frozen:
            params += [p for p in self.SpecMamba.parameters() if p.grad is not None]
        if not params:
            return True
        total_norm = torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)
        if not torch.isfinite(total_norm):
            for p in params:
                p.grad = None
            return False
        return True

    def optimize_parameters(self, current_step: int):
        if self.paradigm == "pcc":
            return self._optimize_pcc(current_step)
        self._update_specmamba_freeze_schedule(current_step)
        self.G_optimizer.zero_grad(set_to_none=True)
        self.SpecMamba_optimizer.zero_grad(set_to_none=True)

        t = self._sample_t(self.L.shape[0], current_step)

        amp_ctx = torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype)
        with amp_ctx:
            fwd = self._forward_train(t)
            total_loss, logs = self._compute_loss(fwd)

        # NaN/Inf guard: skip the whole step if the loss is already non-finite,
        # so a single bad batch cannot poison the weights with nan and stall training.
        if not torch.isfinite(total_loss):
            self._nan_skip_count += 1
            logs["loss"] = total_loss.detach()
        elif self._use_scaler:
            self.scaler.scale(total_loss).backward()
            # Unscale before clipping so the norm is computed on real gradients.
            self.scaler.unscale_(self.G_optimizer)
            if not self._specmamba_is_frozen:
                self.scaler.unscale_(self.SpecMamba_optimizer)
            if self.clip_grad_norm is not None:
                self._clip_grads()
            self.scaler.step(self.G_optimizer)
            if not self._specmamba_is_frozen:
                self.scaler.step(self.SpecMamba_optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            grads_ok = self._clip_grads() if self.clip_grad_norm is not None else True
            if grads_ok:
                self.G_optimizer.step()
                if not self._specmamba_is_frozen:
                    self.SpecMamba_optimizer.step()
            else:
                self._nan_skip_count += 1

        # cache results for visuals
        self.E = torch.cat([self.H[:, :1] * 0, fwd["H_hat_rgb"].detach()], dim=1)  # 4-ch quaternion (zero-pad)
        self.M_VAU = fwd["spec_prob"].detach()

        for k, v in logs.items():
            self.log_dict[k] = float(v.item()) if torch.is_tensor(v) else float(v)
        self.log_dict["SpecMamba_frozen"] = float(self._specmamba_is_frozen)
        self.log_dict["SpecMamba_lr"] = float(self.SpecMamba_optimizer.param_groups[0]["lr"])
        self.log_dict["SpecMamba_size"] = float(self.M_spec.shape[-1])
        self.log_dict["nan_skips"] = float(self._nan_skip_count)

        if self.opt_train["E_decay"] > 0:
            self.update_E(self.opt_train["E_decay"])

    # -----------------------------------------------------------------
    # EMA — vectorised
    # -----------------------------------------------------------------

    def _copy_G_into_E(self):
        src = dict(self.get_bare_model(self.netG).state_dict())
        dst = self.get_bare_model(self.netE).state_dict()
        for k in dst:
            dst[k].copy_(src[k])

    def update_E(self, decay: float = 0.999):
        # Fused multiply/add over parameter tensors.
        src = self.get_bare_model(self.netG)
        dst = self.get_bare_model(self.netE)
        with torch.no_grad():
            src_params = list(src.parameters())
            dst_params = list(dst.parameters())
            torch._foreach_mul_(dst_params, decay)
            torch._foreach_add_(dst_params, src_params, alpha=1.0 - decay)

    # -----------------------------------------------------------------
    # PCC — Physically-Controlled Consistency Training (single-stage)
    # -----------------------------------------------------------------

    def _random_blobs(self, B, H, W, device):
        """Smooth random positive specular-magnitude field beta, shape (B,1,H,W)."""
        ys = torch.linspace(0.0, 1.0, H, device=device)
        xs = torch.linspace(0.0, 1.0, W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        beta = torch.zeros(B, 1, H, W, device=device)
        lo, hi = int(self.pcc_blobs[0]), int(self.pcc_blobs[1])
        amin, amax = float(self.pcc_beta_range[0]), float(self.pcc_beta_range[1])
        for b in range(B):
            for _ in range(random.randint(lo, hi)):
                cy, cx = random.random(), random.random()
                sy, sx = random.uniform(0.05, 0.25), random.uniform(0.05, 0.25)
                amp = random.uniform(amin, amax)
                g = amp * torch.exp(-(((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
                beta[b, 0] = torch.maximum(beta[b, 0], g)
        return beta

    def _synthesize_highlight(self, D):
        """Add a physically-valid (rank-1, illuminant-colored, clipped) highlight
        onto clean diffuse D. Returns (I_syn, mask, sat), each (B,*,H,W):
        mask = highlight support, sat = EXACT clipped/saturated region (known a priori)."""
        B, _, H, W = D.shape
        tint = (torch.rand(B, 3, 1, 1, device=D.device) - 0.5) * 0.2     # illuminant tint in [-0.1, 0.1]
        gamma = torch.ones(B, 3, 1, 1, device=D.device) + tint
        gamma = gamma / gamma.amax(dim=1, keepdim=True).clamp_min(1e-6)  # rank-1 color, max channel = 1
        beta = self._random_blobs(B, H, W, D.device)                     # (B,1,H,W) magnitude
        raw = D + beta * gamma                                            # pre-clip image
        I_syn = raw.clamp(0.0, 1.0)                                       # additive + clip
        mask = (beta > 0.05).float()                                     # highlight support
        sat = (raw.amax(dim=1, keepdim=True) > 1.0).float()             # exact saturated region
        return I_syn, mask, sat

    def _pcc_mask_loss(self, mask_logits_spec, target_long, keep):
        """CE + Dice toward a binary target at SpecMamba resolution.
        keep: (B,) per-sample weight (e.g. has_mask)."""
        ce_per = F.cross_entropy(mask_logits_spec, target_long, reduction="none").mean(dim=(1, 2))
        denom = keep.sum().clamp_min(1.0)
        loss_ce = (ce_per * keep).sum() / denom
        probs = F.softmax(mask_logits_spec, dim=1).float()
        target_oh = F.one_hot(target_long, 2).permute(0, 3, 1, 2).float()
        inter = (probs * target_oh).sum(dim=(1, 2, 3))
        sets = probs.sum(dim=(1, 2, 3)) + target_oh.sum(dim=(1, 2, 3))
        dice_per = 1.0 - (2.0 * inter + 1e-6) / (sets + 1e-6)
        loss_dice = (dice_per * keep).sum() / denom
        return loss_ce + loss_dice

    def _forward_pcc(self):
        """Build the highlight family and restore all members in ONE batched
        forward (DDP-safe: a single forward per backward), then compute PCC losses."""
        H_rgb = self.H[:, 1:]                       # clean diffuse GT (B,3,H,W)
        L = self.L                                  # real input with highlight (B,3,H,W)
        spec_size = self._specmamba_size(H_rgb.shape[-2:])
        B, _, Hh, Ww = L.shape

        images = [L]                                # member 0 is always the REAL input
        masks = [None]
        sats = [None]                               # exact saturation map per synthetic member
        for _ in range(max(0, self.pcc_n - 1)):
            I_syn, m, s = self._synthesize_highlight(H_rgb)
            images.append(I_syn)
            masks.append(m)
            sats.append(s)
        n = len(images)

        # ---- single batched restoration over the whole family ----
        I_all = torch.cat(images, dim=0)                                  # (n*B,3,H,W)
        t = torch.full((n * B,), int(self.T), device=self.device, dtype=torch.long)
        t_emb = self.t_embed(t)
        I_spec = F.interpolate(I_all, size=spec_size, mode="bilinear", align_corners=False)
        if self.pcc_lam_prompt > 0:
            mask_logits_spec, spec_aux = self.SpecMamba(I_spec, return_aux=True)   # (n*B,2,hs,ws)
        else:
            mask_logits_spec = self.SpecMamba(I_spec)                     # (n*B,2,hs,ws)
            spec_aux = None
        mask_logits = F.interpolate(mask_logits_spec, size=(Hh, Ww), mode="bilinear", align_corners=False)
        spec_prob = mask_logits.softmax(dim=1)[:, 1:2]                    # (n*B,1,H,W)
        # Detach mask into the restorer: SpecMamba is supervised only by L_mask.
        q_out = self.netG(torch.cat([spec_prob.detach(), I_all], dim=1), t_emb, mask=spec_prob.detach())
        D_all = q_out[:, 1:]                                             # (n*B,3,H,W)

        D_hats = list(D_all.chunk(n, dim=0))
        spec_chunks = list(spec_prob.chunk(n, dim=0))
        mls_chunks = list(mask_logits_spec.chunk(n, dim=0))

        # ---- mask supervision (GT mask for the real member, known mask for synthetic) ----
        loss_mask = H_rgb.new_zeros(())
        for idx in range(n):
            if idx == 0:
                target = self.M_spec                              # GT mask (long, spec size)
                keep = self.has_mask
            else:
                m_spec = F.interpolate(masks[idx], size=spec_size, mode="nearest")[:, 0]
                target = (m_spec >= 0.5).long()                   # known synthetic mask
                keep = torch.ones(B, device=self.device)
            loss_mask = loss_mask + self._pcc_mask_loss(mls_chunks[idx], target, keep)
        loss_mask = loss_mask / n

        # reflectance-prompt weak supervision (diversity + physics-pseudo routing)
        if self.pcc_lam_prompt > 0 and spec_aux is not None:
            loss_prompt, _ = reflectance_prompt_loss(spec_aux["prompt_weight"], spec_aux["visual_prior"])
        else:
            loss_prompt = H_rgb.new_zeros(())

        # ---- supervised reconstruction: weight the REAL sample above synthetic ones ----
        # (synthetic highlights are easier; don't let 3:1 synthetic dominate real training)
        loss_sup_real = F.l1_loss(D_hats[0], H_rgb)
        if n > 1:
            loss_sup_syn = sum(F.l1_loss(D_hats[i], H_rgb) for i in range(1, n)) / (n - 1)
        else:
            loss_sup_syn = H_rgb.new_zeros(())
        loss_sup = self.pcc_w_real * loss_sup_real + self.pcc_w_syn * loss_sup_syn

        # ---- saturation-weighted recon: EXACT clip mask for synthetic, heuristic for real ----
        H_tile = H_rgb.repeat(n, 1, 1, 1)
        c_real = torch.sigmoid(30.0 * (self.L.amax(dim=1, keepdim=True) - 0.98))   # (B,1,H,W)
        c_sat = torch.cat([c_real] + [sats[i] for i in range(1, n)], dim=0)        # (n*B,1,H,W)
        loss_sat = (c_sat * (D_all - H_tile).abs()).sum() / c_sat.sum().clamp_min(1.0)

        # ---- specular-invariance consistency: synthetic outputs follow the REAL one ----
        # Stop-gradient teacher (real member) prevents collusion/collapse and over-smoothing.
        if n > 1:
            teacher = D_hats[0].detach()
            loss_cons = sum(F.l1_loss(D_hats[i], teacher) for i in range(1, n)) / (n - 1)
        else:
            loss_cons = H_rgb.new_zeros(())
        first_spec_prob = spec_chunks[0]

        total = (
            self.pcc_lam_sup * loss_sup
            + self.pcc_lam_sat * loss_sat
            + self.pcc_lam_cons * loss_cons
            + self.pcc_lam_mask * loss_mask
            + self.pcc_lam_prompt * loss_prompt
        )
        logs = {
            "loss": total.detach(),
            "L_sup": loss_sup.detach(),
            "L_sat": loss_sat.detach(),
            "L_cons": loss_cons.detach(),
            "L_mask": loss_mask.detach(),
            "L_prompt": loss_prompt.detach(),
        }
        # cache visuals from the real member for logging/eval-image dumps
        self.E = torch.cat([self.H[:, :1] * 0, D_hats[0].detach()], dim=1)
        self.M_VAU = first_spec_prob.detach()
        return total, logs

    def _optimize_pcc(self, current_step: int):
        self.G_optimizer.zero_grad(set_to_none=True)
        self.SpecMamba_optimizer.zero_grad(set_to_none=True)

        amp_ctx = torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype)
        with amp_ctx:
            total_loss, logs = self._forward_pcc()

        if not torch.isfinite(total_loss):
            self._nan_skip_count += 1
            logs["loss"] = total_loss.detach()
        elif self._use_scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.G_optimizer)
            self.scaler.unscale_(self.SpecMamba_optimizer)
            if self.clip_grad_norm is not None:
                self._clip_grads()
            self.scaler.step(self.G_optimizer)
            self.scaler.step(self.SpecMamba_optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            grads_ok = self._clip_grads() if self.clip_grad_norm is not None else True
            if grads_ok:
                self.G_optimizer.step()
                self.SpecMamba_optimizer.step()
            else:
                self._nan_skip_count += 1

        for k, v in logs.items():
            self.log_dict[k] = float(v.item()) if torch.is_tensor(v) else float(v)
        self.log_dict["nan_skips"] = float(self._nan_skip_count)

        if self.opt_train["E_decay"] > 0:
            self.update_E(self.opt_train["E_decay"])

    @torch.no_grad()
    def _test_pcc(self):
        """Single-pass restoration for eval (EMA net if available)."""
        infer_net = self.netE if (self.opt_train.get("E_decay", 0) > 0 and hasattr(self, "netE")) else self.netG
        infer_net.eval()
        self.SpecMamba.eval()
        self.t_embed.eval()

        I = self.L
        B, _, H, W = I.shape
        t = torch.full((B,), int(self.T), device=self.device, dtype=torch.long)
        t_emb = self.t_embed(t)
        spec_size = self._specmamba_size((H, W))
        I_spec = F.interpolate(I, size=spec_size, mode="bilinear", align_corners=False)
        mask_logits_spec = self.SpecMamba(I_spec)
        mask_logits = F.interpolate(mask_logits_spec, size=(H, W), mode="bilinear", align_corners=False)
        spec_prob = mask_logits.softmax(dim=1)[:, 1:2]
        q_out = infer_net(torch.cat([spec_prob, I], dim=1), t_emb, mask=spec_prob)
        D_hat = q_out[:, 1:]

        self.E = torch.cat([self.H[:, :1] * 0, D_hat], dim=1)
        self.M_VAU = spec_prob

        self.netG.train()
        self.SpecMamba.train()
        self.t_embed.train()

    # -----------------------------------------------------------------
    # Inference (DDIM-style sampling)
    # -----------------------------------------------------------------

    @torch.no_grad()
    def test(self):
        if self.paradigm == "pcc":
            return self._test_pcc()
        # Use EMA model for inference if available — typically 0.2-0.5 dB better.
        infer_net = self.netE if (self.opt_train.get("E_decay", 0) > 0 and hasattr(self, "netE")) else self.netG
        infer_net.eval()
        self.SpecMamba.eval()
        self.t_embed.eval()

        T = self.T
        steps = max(1, min(self.infer_steps, T))
        ts = torch.linspace(T, 0, steps + 1, device=self.device).round().long()

        B, _, H, W = self.L.shape
        x = self.L                                    # x_T = L
        last_spec = None
        H_hat = self.L                                # safe default if steps==0

        for i in range(steps):
            t_now = ts[i].expand(B)
            t_next = ts[i + 1].expand(B)
            alpha_t = self._alpha_view(t_now)
            alpha_n = self._alpha_view(t_next)

            t_emb = self.t_embed(t_now)
            # SpecMamba sees the current x at the configured size, matching training.
            spec_size = self._specmamba_size((H, W))
            x_spec = F.interpolate(x, size=spec_size, mode="bilinear", align_corners=False)
            mask_logits_spec = self.SpecMamba(x_spec)
            mask_logits = F.interpolate(mask_logits_spec, size=(H, W), mode="bilinear", align_corners=False)
            spec_prob = mask_logits.softmax(dim=1)[:, 1:2]
            last_spec = spec_prob

            q_in = torch.cat([spec_prob, x], dim=1)
            q_out = infer_net(q_in, t_emb, mask=spec_prob)
            if self.predict_residual:
                H_hat = (x - q_out[:, 1:]).clamp(0, 1)
            else:
                H_hat = q_out[:, 1:]

            # Cold Diffusion (Bansal et al. 2022, Alg. 2) sampling step:
            #   x_{t-1} = x_t - D(H_hat, t) + D(H_hat, t-1)
            # with corruption D(H, t) = H + alpha_t * (L - H), this simplifies to
            #   x = x - (alpha_t - alpha_n) * (L - H_hat)
            # which is provably more stable than vanilla DDIM for deterministic
            # (non-Gaussian) corruption processes.
            x = x - (alpha_t - alpha_n) * (self.L - H_hat)

        self.E = torch.cat([self.H[:, :1] * 0, H_hat], dim=1)
        self.M_VAU = last_spec

        self.netG.train()
        self.SpecMamba.train()
        self.t_embed.train()

    # -----------------------------------------------------------------
    # Visuals / logging
    # -----------------------------------------------------------------

    def current_log(self):
        return self.log_dict

    def current_visuals(self, need_H: bool = True):
        out = OrderedDict()
        out["L"] = self.L.detach()[0].float().cpu()
        out["E"] = self.E[:, 1:].detach()[0].float().cpu()
        out["M"] = self.M_VAU.detach()[0].float().cpu()
        if need_H:
            out["H"] = self.H[:, 1:].detach()[0].float().cpu()
        return out

    def current_results(self, need_H: bool = True):
        out = OrderedDict()
        out["L"] = self.L.detach().float().cpu()
        out["E"] = self.E[:, 1:].detach().float().cpu()
        out["M"] = self.M_VAU.detach()[0].float().cpu()
        if need_H:
            out["H"] = self.H[:, 1:].detach().float().cpu()
        return out

    # -----------------------------------------------------------------
    # Info helpers (kept for compatibility with main_*.py logging)
    # -----------------------------------------------------------------

    def print_network(self):
        print(self.describe_network(self.netG))

    def print_params(self):
        print(self.describe_params(self.netG))

    def info_network(self):
        return self.describe_network(self.netG)

    def info_params(self):
        return self.describe_params(self.netG)
