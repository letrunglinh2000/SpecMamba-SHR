"""
Diffusion-style training utilities for SpecMamba + Qformer.

Provides:
  - SinusoidalTimeEmbed: sinusoidal timestep embedding
  - TimeFiLM: zero-initialised channel-wise FiLM modulation (identity at init)
  - AlphaSchedule: cosine schedule for the physics-residual diffusion process
  - DiffusionQformer: wrapper that orchestrates Qformer's submodules and
    injects TimeFiLM between encoder/decoder levels
  - DiffusionSpecMamba: wrapper that adds a zero-initialised time-to-prompt
    projection so the existing PromptFiLM machinery becomes time-conditioned

Design goals:
  - Loading pretrained Qformer / SpecMamba weights works unchanged
    (strict=False for the new FiLM / time-embed parameters).
  - At init, FiLM is identity, so the model reproduces pretrained behaviour
    bit-identically at t = T.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.quaternion_ops as core_qnn


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "time-embed dim must be even"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: (B,) float or long
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        emb = t.float()[:, None] * freqs[None]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)


class TimeEmbedMLP(nn.Module):
    """Sinusoidal -> MLP, used as the shared time conditioning vector."""

    def __init__(self, dim: int = 128, hidden: int = 256):
        super().__init__()
        self.sinu = SinusoidalTimeEmbed(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinu(t))


# ---------------------------------------------------------------------------
# Zero-initialised FiLM (identity at init)
# ---------------------------------------------------------------------------

class TimeFiLM(nn.Module):
    def __init__(self, channels: int, t_dim: int):
        super().__init__()
        self.proj = nn.Linear(t_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        gb = self.proj(t_emb)[:, :, None, None]
        gamma, beta = gb.chunk(2, dim=1)
        return feat * (1.0 + gamma) + beta


class MaskFiLM(nn.Module):
    """Per-pixel FiLM modulation driven by the SpecMamba mask.

    Resizes the mask to the feature resolution, then projects (1 -> 2C) so it
    can scale/shift the features spatially. Zero-init so identity at start.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(1, 2 * channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape[-2:] != feat.shape[-2:]:
            mask = F.interpolate(mask, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        gb = self.proj(mask)
        gamma, beta = gb.chunk(2, dim=1)
        return feat * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# Cosine alpha schedule for L = H + alpha * (L - H)
# ---------------------------------------------------------------------------

class AlphaSchedule:
    """Cosine schedule with alpha_0 ≈ 0, alpha_T = 1."""

    def __init__(self, T: int = 8, s: float = 0.008):
        self.T = T
        ts = torch.arange(T + 1, dtype=torch.float32) / T
        alphas = 1.0 - torch.cos((ts + s) / (1 + s) * math.pi / 2) ** 2
        alphas = alphas / alphas[-1]
        self.alphas = alphas  # shape (T+1,), alphas[0]=0, alphas[T]=1

    def to(self, device):
        self.alphas = self.alphas.to(device)
        return self

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return self.alphas[t.long()]


# ---------------------------------------------------------------------------
# Wrapped Qformer with time conditioning
# ---------------------------------------------------------------------------

class DiffusionQformer(nn.Module):
    """
    Wraps a pretrained Qformer and inserts TimeFiLM between its U-Net stages.

    Forward path mirrors Qformer.forward() in models/Qformer.py, with FiLM
    applied after each encoder/decoder stage. All new FiLM modules are
    zero-initialised so the output equals the pretrained Qformer at init.
    """

    def __init__(
        self,
        base_qformer: nn.Module,
        t_dim: int = 128,
        use_mask_film: bool = True,
        predict_residual: bool = True,
    ):
        super().__init__()
        self.base = base_qformer
        self.use_mask_film = use_mask_film
        self.predict_residual = predict_residual
        # Qformer uses QuaternionConv whose `out_channels` is the quaternion-logical
        # count; the feature tensor has 4x as many real channels.
        dim = base_qformer.patch_embed.proj.out_channels * 4  # level-1 dim (e.g. 80)
        d1, d2, d3, d4 = dim, dim * 2, dim * 4, dim * 8
        # Decoder dims: dec3 has d3 channels, dec2 has d2, dec1 has d2 (no 1x1 reduce on level1)
        self.film_e1 = TimeFiLM(d1, t_dim)
        self.film_e2 = TimeFiLM(d2, t_dim)
        self.film_e3 = TimeFiLM(d3, t_dim)
        self.film_latent = TimeFiLM(d4, t_dim)
        self.film_d3 = TimeFiLM(d3, t_dim)
        self.film_d2 = TimeFiLM(d2, t_dim)
        self.film_d1 = TimeFiLM(d2, t_dim)  # decoder_level1 keeps d2 channels
        if use_mask_film:
            self.mfilm_e1 = MaskFiLM(d1)
            self.mfilm_e2 = MaskFiLM(d2)
            self.mfilm_e3 = MaskFiLM(d3)
            self.mfilm_latent = MaskFiLM(d4)
            self.mfilm_d3 = MaskFiLM(d3)
            self.mfilm_d2 = MaskFiLM(d2)
            self.mfilm_d1 = MaskFiLM(d2)

    def forward(
        self,
        inp_img: torch.Tensor,
        t_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = self.base
        use_m = self.use_mask_film and mask is not None
        x = b.patch_embed(inp_img)

        e1 = b.encoder_level1(x)
        e1 = self.film_e1(e1, t_emb)
        if use_m: e1 = self.mfilm_e1(e1, mask)

        e2 = b.encoder_level2(b.down1_2(e1))
        e2 = self.film_e2(e2, t_emb)
        if use_m: e2 = self.mfilm_e2(e2, mask)

        e3 = b.encoder_level3(b.down2_3(e2))
        e3 = self.film_e3(e3, t_emb)
        if use_m: e3 = self.mfilm_e3(e3, mask)

        latent = b.latent(b.down3_4(e3))
        latent = self.film_latent(latent, t_emb)
        if use_m: latent = self.mfilm_latent(latent, mask)

        d3 = b.up4_3(latent)
        d3 = core_qnn.quater_concate_old(d3, e3)
        d3 = b.reduce_chan_level3(d3)
        d3 = b.decoder_level3(d3)
        d3 = self.film_d3(d3, t_emb)
        if use_m: d3 = self.mfilm_d3(d3, mask)

        d2 = b.up3_2(d3)
        d2 = core_qnn.quater_concate_old(d2, e2)
        d2 = b.reduce_chan_level2(d2)
        d2 = b.decoder_level2(d2)
        d2 = self.film_d2(d2, t_emb)
        if use_m: d2 = self.mfilm_d2(d2, mask)

        d1 = b.up2_1(d2)
        d1 = core_qnn.quater_concate_old(d1, e1)
        d1 = b.decoder_level1(d1)
        d1 = self.film_d1(d1, t_emb)
        if use_m: d1 = self.mfilm_d1(d1, mask)

        d1 = b.refinement(d1)

        if self.predict_residual:
            # Output IS predicted specular layer; downstream computes
            # H_hat = (L - S).clamp(0, 1). Do not add inp_img here.
            if b.dual_pixel_task:
                d1 = d1 + b.skip_conv(x)
            out = b.output(d1)
        else:
            if b.dual_pixel_task:
                d1 = d1 + b.skip_conv(x)
                out = b.output(d1)
            else:
                out = b.output(d1) + inp_img

        return out


# ---------------------------------------------------------------------------
# Wrapped AnyIR with time conditioning (ablation backbone)
# ---------------------------------------------------------------------------

class DiffusionAnyIR(nn.Module):
    """
    Wraps an AnyIR backbone with:
      - TimeFiLM at every U-Net stage (time conditioning),
      - optional MaskFiLM at every stage (multi-scale mask injection),
      - optional residual-prediction mode (output = predicted specular layer,
        not added to input; consumer computes H_hat = (L - S).clamp(0, 1)).

    All FiLM modules are zero-init -> identity at start.
    """

    def __init__(
        self,
        base_anyir: nn.Module,
        t_dim: int = 128,
        use_mask_film: bool = True,
        predict_residual: bool = True,
    ):
        super().__init__()
        self.base = base_anyir
        self.use_mask_film = use_mask_film
        self.predict_residual = predict_residual
        dim = base_anyir.patch_embed.proj.out_channels
        d1, d2, d3, d4 = dim, dim * 2, dim * 4, dim * 8
        del d4  # latent is reduced to d3 before FiLM
        self.film_e1 = TimeFiLM(d1, t_dim)
        self.film_e2 = TimeFiLM(d2, t_dim)
        self.film_e3 = TimeFiLM(d3, t_dim)
        self.film_latent = TimeFiLM(d3, t_dim)
        self.film_d3 = TimeFiLM(d3, t_dim)
        self.film_d2 = TimeFiLM(d2, t_dim)
        self.film_d1 = TimeFiLM(d2, t_dim)
        if use_mask_film:
            self.mfilm_e1 = MaskFiLM(d1)
            self.mfilm_e2 = MaskFiLM(d2)
            self.mfilm_e3 = MaskFiLM(d3)
            self.mfilm_latent = MaskFiLM(d3)
            self.mfilm_d3 = MaskFiLM(d3)
            self.mfilm_d2 = MaskFiLM(d2)
            self.mfilm_d1 = MaskFiLM(d2)

    def forward(
        self,
        inp_img: torch.Tensor,
        t_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = self.base
        use_m = self.use_mask_film and mask is not None
        x = b.patch_embed(inp_img)

        e1 = b.encoder_level1(x)
        e1 = self.film_e1(e1, t_emb)
        if use_m: e1 = self.mfilm_e1(e1, mask)

        e2 = b.encoder_level2(b.down1_2(e1))
        e2 = self.film_e2(e2, t_emb)
        if use_m: e2 = self.mfilm_e2(e2, mask)

        e3 = b.encoder_level3(b.down2_3(e2))
        e3 = self.film_e3(e3, t_emb)
        if use_m: e3 = self.mfilm_e3(e3, mask)

        latent = b.latent(b.down3_4(e3))
        latent = b.reduce_dim_level3(latent)
        latent = self.film_latent(latent, t_emb)
        if use_m: latent = self.mfilm_latent(latent, mask)

        d3 = b.up4_3(latent)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = b.reduce_chan_level3(d3)
        d3 = b.decoder_level3(d3)
        d3 = self.film_d3(d3, t_emb)
        if use_m: d3 = self.mfilm_d3(d3, mask)

        d2 = b.up3_2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = b.reduce_chan_level2(d2)
        d2 = b.decoder_level2(d2)
        d2 = self.film_d2(d2, t_emb)
        if use_m: d2 = self.mfilm_d2(d2, mask)

        d1 = b.up2_1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = b.decoder_level1(d1)
        d1 = self.film_d1(d1, t_emb)
        if use_m: d1 = self.mfilm_d1(d1, mask)

        d1 = b.refinement(d1)
        if self.predict_residual:
            # Output IS the predicted specular layer; downstream computes
            # H_hat = (L - S).clamp(0, 1). Do not add inp_img here.
            out = b.output(d1)
        else:
            out = b.output(d1) + inp_img
        return out


# ---------------------------------------------------------------------------
# Wrapped SpecMamba with time conditioning
# ---------------------------------------------------------------------------

class DiffusionSpecMamba(nn.Module):
    """
    Wraps a pretrained Specmamba. The mask is a property of the current image
    content (specular vs diffuse), so it does not depend on the diffusion
    timestep — no time conditioning is injected here.
    """

    def __init__(self, base_specmamba: nn.Module, t_dim: int = 128):
        super().__init__()
        del t_dim  # kept for backward-compat signature
        self.base = base_specmamba

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None, return_aux: bool = False):
        del t_emb  # SpecMamba is time-agnostic
        b = self.base
        input_size = x.shape[-2:]
        prior = b.visual_prior(x)
        prompt_embed, prompt_weight = b.prompt_bank(prior)

        x1 = b.stem(x)
        x1 = b.prompt_film1(x1, prompt_embed)
        if b.prompt_gate1 is not None:
            x1, _ = b.prompt_gate1(x1, prior)
        x1 = b.enc1(x1)

        x2 = b.down1(x1)
        x2 = b.prompt_film2(x2, prompt_embed)
        if b.prompt_gate2 is not None:
            x2, _ = b.prompt_gate2(x2, prior)
        x2 = b.enc2(x2)

        x3 = b.down2(x2)
        x3 = b.prompt_film3(x3, prompt_embed)
        if b.prompt_gate3 is not None:
            x3, _ = b.prompt_gate3(x3, prior)
        x3 = b.enc3(x3)

        x4 = b.down3(x3)
        x4 = b.prompt_film4(x4, prompt_embed)
        x4 = b.bottleneck(x4)

        y3 = b.up3(x4, x3)
        y3 = b.prompt_film3(y3, prompt_embed)
        y3 = b.dec3(y3)

        y2 = b.up2(y3, x2)
        y2 = b.prompt_film2(y2, prompt_embed)
        y2 = b.dec2(y2)

        y1 = b.up1(y2, x1)
        y1 = b.prompt_film1(y1, prompt_embed)
        y1 = b.dec1(y1)

        logits = b.output(y1)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        if return_aux:
            return logits, {"visual_prior": prior, "prompt_weight": prompt_weight}
        return logits


# ---------------------------------------------------------------------------
# Helper: load pretrained state-dict into a wrapped model
# ---------------------------------------------------------------------------

def load_pretrained_into_wrapper(wrapper: nn.Module, state_dict: dict) -> None:
    """
    Load a state-dict into a wrapper whose base submodule is `base`.

    Two checkpoint shapes are supported:
      (a) Vanilla (pretrained backbone), keys like  "patch_embed.proj.weight"
      (b) Resume (wrapper format),        keys like  "base.patch_embed.proj..."

    Missing keys for the new FiLM / time modules are allowed; everything else
    must match.
    """
    if not state_dict:
        return
    sample_key = next(iter(state_dict.keys()))
    needs_remap = not sample_key.startswith("base.") and "film_" not in sample_key and "time_to_prompt" not in sample_key
    if needs_remap:
        state_dict = {f"base.{k}": v for k, v in state_dict.items()}

    # Strip keys that no longer exist in the model (e.g. time_to_prompt after
    # SpecMamba was made time-agnostic). Filter before load to avoid the
    # strict=False unexpected-keys path raising.
    obsolete_tokens = ("time_to_prompt",)
    state_dict = {
        k: v for k, v in state_dict.items()
        if not any(tok in k for tok in obsolete_tokens)
    }

    missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
    allowed_missing_tokens = ("film_", "time_to_prompt")
    bad_missing = [
        k for k in missing
        if not any(tok in k for tok in allowed_missing_tokens)
    ]
    if bad_missing:
        raise RuntimeError(f"Unexpected missing keys: {bad_missing[:10]} ...")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:10]} ...")
