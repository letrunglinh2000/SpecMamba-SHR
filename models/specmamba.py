# import torch
# import torch.nn as nn
# import math
# from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
# from timm.models.vision_transformer import Mlp
# import torch.nn.functional as F
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
# from einops import rearrange, repeat
# from typing import Dict, Tuple, Optional

# def window_partition(x, window_size):
#     """
#     Args:
#         x: (B, C, H, W)
#         window_size: window size
#         h_w: Height of window
#         w_w: Width of window
#     Returns:
#         local window features (num_windows*B, window_size*window_size, C)
#     """
#     B, C, H, W = x.shape
#     x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
#     windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
#     return windows


# def window_reverse(windows, window_size, H, W):
#     """
#     Args:
#         windows: local window features (num_windows*B, window_size, window_size, C)
#         window_size: Window size
#         H: Height of image
#         W: Width of image
#     Returns:
#         x: (B, C, H, W)
#     """
#     B = int(windows.shape[0] / (H * W / window_size / window_size))
#     x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
#     x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
#     return x

# class PatchEmbed(nn.Module):
#     """
#     Patch embedding block"
#     """

#     def __init__(self, in_chans=3, in_dim=64, dim=96):
#         """
#         Args:
#             in_chans: number of input channels.
#             dim: feature size dimension.
#         """
#         # in_dim = 1
#         super().__init__()
#         self.proj = nn.Identity()
#         self.conv_down = nn.Sequential(
#             nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(in_dim, eps=1e-4),
#             nn.ReLU(),
#             nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(dim, eps=1e-4),
#             nn.ReLU()
#             )

#     def forward(self, x):
#         x = self.proj(x)
#         x = self.conv_down(x)
#         return x


# class ConvGNAct(nn.Module):
#     """Small stable conv block."""
#     def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None):
#         super().__init__()
#         if p is None:
#             p = k // 2
#         self.block = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
#             LayerNorm2d(out_ch),
#             nn.GELU()
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.block(x)
    
# class SpecularAdaptiveBandSplitter(nn.Module):
#     """
#     Highlight-aware spectral decomposition block (your outstanding module).
#     Fixed: gain head shape bug + minor stability improvements.
#     """
#     def __init__(
#         self,
#         dim: int,
#         reduction: int = 4,
#         cutoff_range_low: Tuple[float, float] = (0.04, 0.14),
#         cutoff_range_mid: Tuple[float, float] = (0.16, 0.34),
#         mask_sharpness: float = 28.0,
#         use_residual: bool = True,
#     ):
#         super().__init__()
#         hidden = max(dim // reduction, 8)
#         self.dim = dim
#         self.cutoff_range_low = cutoff_range_low
#         self.cutoff_range_mid = cutoff_range_mid
#         self.mask_sharpness = mask_sharpness
#         self.use_residual = use_residual

#         # 1) highlight prior predictor
#         self.prior_head = nn.Sequential(
#             ConvGNAct(dim, hidden, 3),
#             nn.Conv2d(hidden, 2, kernel_size=1, bias=True)
#         )

#         # 2) global cutoff predictor
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.cutoff_head = nn.Sequential(
#             nn.Conv2d(dim, hidden, 1, bias=False),
#             nn.GELU(),
#             nn.Conv2d(hidden, 2, 1, bias=True)
#         )

#         # 3) dynamic gain predictor
#         self.gain_head = nn.Sequential(
#             nn.Conv2d(dim + 2, hidden, 1, bias=False),
#             nn.GELU(),
#             nn.Conv2d(hidden, 3 * dim, 1, bias=True)   # 3 bands
#         )

#         # 4) band refinement
#         self.low_refine = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
#             nn.Conv2d(dim, dim, 1, bias=False),
#             LayerNorm2d(dim),
#             nn.GELU(),
#         )
#         self.mid_refine = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
#             nn.Conv2d(dim, dim, 1, bias=False),
#             LayerNorm2d(dim),
#             nn.GELU(),
#         )
#         self.high_refine = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
#             nn.Conv2d(dim, dim, 1, bias=False),
#             LayerNorm2d(dim),
#             nn.GELU(),
#         )

#         # 5) cross-band compensation
#         self.high_to_mid = nn.Sequential(
#             nn.Conv2d(dim + 2, dim, 1, bias=False),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
#         )
#         self.mid_to_low = nn.Sequential(
#             nn.Conv2d(dim + 1, dim, 1, bias=False),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
#         )
#         self.low_to_high = nn.Sequential(
#             nn.Conv2d(dim + 2, dim, 1, bias=False),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
#         )

#         # 6) final fusion
#         self.out_proj = nn.Sequential(
#             nn.Conv2d(dim * 3, dim, 1, bias=False),
#             LayerNorm2d(dim),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, padding=1, bias=False),
#         )

#         self.res_scale = nn.Parameter(torch.tensor(1.0))
#         self.high_boost = nn.Parameter(torch.tensor(0.75))
#         self.boundary_boost = nn.Parameter(torch.tensor(0.50))
#         self.low_suppress = nn.Parameter(torch.tensor(0.20))
#         self.mid_comp_scale = nn.Parameter(torch.tensor(0.30))
#         self.high_comp_scale = nn.Parameter(torch.tensor(0.20))

#     @staticmethod
#     def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
#         kx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], device=x.device, dtype=x.dtype).unsqueeze(0)
#         ky = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], device=x.device, dtype=x.dtype).unsqueeze(0)
#         gx = F.conv2d(x, kx, padding=1)
#         gy = F.conv2d(x, ky, padding=1)
#         edge = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
#         return edge

#     @staticmethod
#     def _normalize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#         b = x.shape[0]
#         x_flat = x.view(b, -1)
#         x_min = x_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
#         x_max = x_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
#         return (x - x_min) / (x_max - x_min + eps)

#     def _predict_priors(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         prior_logits = self.prior_head(x)
#         spec_map = torch.sigmoid(prior_logits[:, 0:1])
#         boundary_seed = torch.sigmoid(prior_logits[:, 1:2])
#         sobel = self._normalize_map(self._sobel_edges(spec_map))
#         boundary_map = torch.clamp(0.5 * boundary_seed + 0.5 * sobel, 0.0, 1.0)
#         return spec_map, boundary_map

#     def _predict_cutoffs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         g = self.global_pool(x)
#         raw = self.cutoff_head(g)
#         low_min, low_max = self.cutoff_range_low
#         mid_min, mid_max = self.cutoff_range_mid
#         low_cutoff = low_min + (low_max - low_min) * torch.sigmoid(raw[:, 0:1])
#         mid_center = mid_min + (mid_max - mid_min) * torch.sigmoid(raw[:, 1:2])
#         mid_center = torch.maximum(mid_center, low_cutoff + 0.06)
#         mid_center = torch.clamp(mid_center, max=0.45)
#         return low_cutoff, mid_center

#     def _build_radial_masks(self, h: int, w: int, device, dtype, low_cutoff, mid_center):
#         fy = torch.fft.fftfreq(h, d=1.0, device=device, dtype=dtype)
#         fx = torch.fft.fftfreq(w, d=1.0, device=device, dtype=dtype)
#         yy, xx = torch.meshgrid(fy, fx, indexing="ij")
#         radius = torch.sqrt(xx.pow(2) + yy.pow(2)).view(1, 1, h, w)

#         mask_low = torch.exp(- (radius / (low_cutoff + 1e-6)).pow(2))
#         mid_width = 0.10 + 0.08 * low_cutoff
#         mask_mid = torch.exp(- ((radius - mid_center) / (mid_width + 1e-6)).pow(2))
#         high_cutoff = torch.clamp(mid_center + 0.10, max=0.48)
#         mask_high = torch.sigmoid((radius - high_cutoff) * self.mask_sharpness)

#         denom = mask_low + mask_mid + mask_high + 1e-8
#         return mask_low / denom, mask_mid / denom, mask_high / denom

#     @staticmethod
#     def _fft_split(x, mask_low, mask_mid, mask_high):
#         x_fft = torch.fft.fft2(x, norm="ortho")
#         low = torch.fft.ifft2(x_fft * mask_low, norm="ortho").real
#         mid = torch.fft.ifft2(x_fft * mask_mid, norm="ortho").real
#         high = torch.fft.ifft2(x_fft * mask_high, norm="ortho").real
#         return low, mid, high

#     def forward(self, x: torch.Tensor, return_aux: bool = False):
#         b, c, h, w = x.shape
#         assert c == self.dim

#         spec_map, boundary_map = self._predict_priors(x)
#         low_cutoff, mid_center = self._predict_cutoffs(x)
#         mask_low, mask_mid, mask_high = self._build_radial_masks(
#             h, w, x.device, x.dtype, low_cutoff, mid_center
#         )

#         low, mid, high = self._fft_split(x, mask_low, mask_mid, mask_high)

#         low = self.low_refine(low)
#         mid = self.mid_refine(mid)
#         high = self.high_refine(high)

#         # Highlight-aware modulation
#         high = high * (1.0 + self.high_boost * spec_map + self.boundary_boost * boundary_map)
#         low = low * (1.0 - self.low_suppress.clamp(0.0, 1.0) * spec_map)

#         # Cross-band compensation
#         mid = mid + self.mid_comp_scale * self.high_to_mid(torch.cat([high, spec_map, boundary_map], dim=1))
#         low = low + self.mid_to_low(torch.cat([mid, spec_map], dim=1))
#         high = high + self.high_comp_scale * self.low_to_high(torch.cat([low, spec_map, boundary_map], dim=1))

#         # Dynamic channel-wise gains (FIXED)
#         gains_input = torch.cat([x, spec_map, boundary_map], dim=1)
#         gains = self.gain_head(gains_input)
#         gains = self.global_pool(gains)                    # (B, 3*dim, 1, 1)
#         gains = gains.view(b, 3, self.dim, 1, 1)           # ← fixed
#         gains = torch.softmax(gains, dim=1)
#         g_low, g_mid, g_high = gains[:, 0], gains[:, 1], gains[:, 2]

#         fused = torch.cat([
#             low * g_low,
#             mid * g_mid,
#             high * g_high
#         ], dim=1)

#         out = self.out_proj(fused)
#         if self.use_residual:
#             out = x + self.res_scale * out

#         if not return_aux:
#             return out

#         aux = {
#             "spec_map": spec_map,
#             "boundary_map": boundary_map,
#             "low_cutoff": low_cutoff,
#             "mid_center": mid_center,
#             "mask_low": mask_low,
#             "mask_mid": mask_mid,
#             "mask_high": mask_high,
#             "low_band": low,
#             "mid_band": mid,
#             "high_band": high,
#         }
#         return out, aux

# #  =====================================================================
# # FINAL FREQUENCY MIXER (drop-in replacement for Attention)
# # =====================================================================
# class FrequencySpecularMixer(nn.Module):
#     """
#     Clean wrapper that uses your outstanding SpecularAdaptiveBandSplitter.
#     Works perfectly with your window_partition / MaskedBlock.
#     """
#     def __init__(self, dim: int, bias: bool = False, layer_idx: int = None):
#         super().__init__()
#         self.dim = dim
#         self.layer_idx = layer_idx

#         self.band_splitter = SpecularAdaptiveBandSplitter(dim=dim)

#         # Final projection back to token space
#         self.out_proj = nn.Linear(dim, dim, bias=bias)

#     def forward(self, x):
#         """
#         x: (B_win, window_size*window_size, dim)   ← from window_partition
#         Returns: same shape
#         """
#         B_win, L, C = x.shape
#         window_size = int(math.sqrt(L))
#         assert window_size * window_size == L, f"Expected square window, got L={L}"

#         # Reshape to spatial
#         x_2d = rearrange(x, "b (h w) c -> b c h w", h=window_size, w=window_size)

#         # Your powerful adaptive frequency splitter
#         out = self.band_splitter(x_2d)                    # (B_win, C, H_win, W_win)

#         # Back to tokens
#         x_out = rearrange(out, "b c h w -> b (h w) c")
#         x_out = self.out_proj(x_out)

#         return x_out

# # ───────────────────────────────────────────────────────────
# # Specular mask predictor (1x1 DW conv + sigmoid)
# # Predicts a soft specular mask from input features, not from GT.
# # ───────────────────────────────────────────────────────────

# class SpecMaskPredictor(nn.Module):
#     def __init__(self, in_channels: int, hidden: int = 64):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, hidden, 1, bias=False)
#         self.norm1 = LayerNorm2d(hidden)
#         self.conv_dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
#         self.norm2 = LayerNorm2d(hidden)
#         self.conv_out = nn.Conv2d(hidden, 1, 1, bias=True)

#     def forward(self, x):
#         x = F.silu(self.norm1(self.conv1(x)))
#         x = F.silu(self.norm2(self.conv_dw(x)))
#         mask = torch.sigmoid(self.conv_out(x))  # Bound to [0, 1]
#         return mask

# class MaskedMambaVisionMixer(nn.Module):
#     """MambaVisionMixer gated by a predicted specular mask.

#     The mask controls the delta (dt) in the selective scan:
#         dt_masked = dt * (1.0 - mask)

#     When mask=1 (specular pixel), dt=0 → no information propagated.
#     When mask=0 (non-specular), dt unchanged → normal propagation.

#     forward(hidden_states, input_features)
#       - hidden_states: (B, L, D) — sequence of hidden states
#       - input_features: (B, C, H, W) — 2D features used to predict the mask
#                          L == H*W
#     """
#     def __init__(
#         self,
#         d_model,
#         d_state=16,
#         d_conv=4,
#         expand=2,
#         dt_rank="auto",
#         dt_min=0.001,
#         dt_max=0.1,
#         dt_init="random",
#         dt_scale=1.0,
#         dt_init_floor=1e-4,
#         conv_bias=True,
#         bias=False,
#         use_fast_path=True,
#         layer_idx=None,
#         device=None,
#         dtype=None,
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.d_model = d_model
#         self.d_state = d_state
#         self.d_conv = d_conv
#         self.expand = expand
#         self.d_inner = int(self.expand * self.d_model)
#         self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
#         self.use_fast_path = use_fast_path
#         self.layer_idx = layer_idx

#         # Mask predictor (1x1 DW conv + sigmoid)
#         self.mask_predictor = SpecMaskPredictor(in_channels=d_model)

#         # Mamba core (same as MambaVisionMixer)
#         self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
#         self.x_proj = nn.Linear(
#             self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
#         )
#         self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
#         dt_init_std = self.dt_rank**-0.5 * dt_scale
#         if dt_init == "constant":
#             nn.init.constant_(self.dt_proj.weight, dt_init_std)
#         elif dt_init == "random":
#             nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
#         else:
#             raise NotImplementedError
#         dt = torch.exp(
#             torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=dt_init_floor)
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             self.dt_proj.bias.copy_(inv_dt)
#         self.dt_proj.bias._no_reinit = True
#         A = repeat(
#             torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=self.d_inner//2,
#         ).contiguous()
#         A_log = torch.log(A)
#         self.A_log = nn.Parameter(A_log)
#         self.A_log._no_weight_decay = True
#         self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
#         self.D._no_weight_decay = True
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
#         self.conv1d_x = nn.Conv1d(
#             in_channels=self.d_inner//2,
#             out_channels=self.d_inner//2,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             groups=self.d_inner//2,
#             **factory_kwargs,
#         )
#         self.conv1d_z = nn.Conv1d(
#             in_channels=self.d_inner//2,
#             out_channels=self.d_inner//2,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             groups=self.d_inner//2,
#             **factory_kwargs,
#         )

#     def forward(self, hidden_states):
#         """
#         Args:
#             hidden_states: (B, L, D) — sequence tokens
#         Returns:
#             out: (B, L, D)
#         """
#         _, seqlen, _ = hidden_states.shape
#         H = W = int(math.sqrt(seqlen))
#         if H * W != seqlen:
#             raise ValueError(f"Expected square token map, got seqlen={seqlen}")

#         # ── Predict specular mask from hidden_states ──
#         hidden_2d = rearrange(hidden_states, "b (h w) d -> b d h w", h=H, w=W)  # (B, C, H, W)
#         mask = self.mask_predictor(hidden_2d)  # (B, 1, H, W)

#         # ── Mamba core ──
#         xz = self.in_proj(hidden_states)
#         xz = rearrange(xz, "b l d -> b d l")
#         x, z = xz.chunk(2, dim=1)
#         A = -torch.exp(self.A_log.float())
#         x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
#         z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
#         x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
#         dt, B_scan, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
#         dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)  # (B, d_inner//2, H*W)

#         # Apply bias and ensure positivity
#         dt_bias = self.dt_proj.bias.float().view(1, -1, 1)
#         dt = dt + dt_bias
#         dt = F.softplus(dt) # make sure dt is positive after adding bias
        
#         # ── Gate dt with specular mask ──
#         # mask: (B, 1, H, W) → (B, 1, H*W) → broadcast over d_inner//2
#         mask_flat = rearrange(mask, "b c h w -> b c (h w)")
#         dt = dt * (1.0 - mask_flat)

#         B_scan = rearrange(B_scan, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
#         C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
#         y = selective_scan_fn(x,
#                               dt,
#                               A,
#                               B_scan,
#                               C,
#                               self.D.float(),
#                               z=None,
#                               delta_bias=None,  # already added above
#                               delta_softplus=False, # dt is already softplus-ed above, no need to apply again in the scan
#                               return_last_state=None)

#         y = torch.cat([y, z], dim=1)
#         y = rearrange(y, "b d l -> b l d")
#         out = self.out_proj(y)
#         return out
       
# class SpecFreMambaBlock(nn.Module):
#     """Main block for feature extraction block in specular highlight removal."""
#     def __init__(
#         self,
#         dim,
#         mixer_type="frequency",  # "frequency" or "mamba"
#         mlp_ratio=4.,
#         drop=0.,
#         drop_path=0.,
#         act_layer=nn.GELU,
#         norm_layer=nn.LayerNorm,
#         Mlp_block=Mlp,
#         layer_scale=None,
#         stage_idx=0,
#     ):
#         super().__init__()
        
#         if mixer_type == "frequency":
#             self.mixer = FrequencySpecularMixer(dim=dim, layer_idx=stage_idx)
#         elif mixer_type == "mamba":
#             self.mixer = MaskedMambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1, layer_idx=stage_idx)
#         else:
#             raise NotImplementedError("Mixer type not implemented.")

#         self.norm1 = norm_layer(dim)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#         use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
#         self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1.0
#         self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1.0

#     def forward(self, x):
#         # Pass features to mixer — MaskedMambaVisionMixer predicts mask from hidden_states
#         x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
#         x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
#         return x
    

# class MaskedMambaVisionLayer(nn.Module):
#     """MambaVisionLayer with masked mixer blocks."""
#     def __init__(
#         self,
#         dim,
#         mixer_type="frequency",  # "frequency" or "mamba"
#         mlp_ratio=4.,
#         drop=0.,
#         drop_path=0.,
#         norm_layer=nn.LayerNorm,
#         Mlp=Mlp,
#         layer_scale=None,
#         downsample=False,
#         window_size=8,
#         stage_idx=0,
#         num_blocks=2,

#     ):
#         super().__init__()
#         self.mixer_type = mixer_type
#         if self.mixer_type == "frequency":
#             self.window_partition = True
#         else:
#             self.window_partition = False

#         self.blocks = nn.ModuleList([
#             SpecFreMambaBlock(
#                 dim=dim,
#                 mixer_type=self.mixer_type,
#                 mlp_ratio=mlp_ratio,
#                 drop=drop,
#                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                 act_layer=nn.GELU,
#                 norm_layer=nn.LayerNorm,
#                 Mlp_block=Mlp,
#                 layer_scale=layer_scale,
#                 stage_idx=stage_idx,
#             )
#             for i in range(num_blocks)
#         ])

#         self.window_size = window_size

#     def forward(self, x):
#         _, _, H, W = x.shape

#         if self.window_partition:
#             pad_r = (self.window_size - W % self.window_size) % self.window_size
#             pad_b = (self.window_size - H % self.window_size) % self.window_size
#             if pad_r > 0 or pad_b > 0:
#                 x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
#                 _, _, Hp, Wp = x.shape
#             else:
#                 Hp, Wp = H, W
#             x = window_partition(x, self.window_size)
#         else:
#             x = rearrange(x, "b c h w -> b (h w) c") # (B, H*W, C)

#         for blk in self.blocks:
#             x = blk(x)
#         if self.window_partition:
#             x = window_reverse(x, self.window_size, Hp, Wp)
#             if pad_r > 0 or pad_b > 0:
#                 x = x[:, :, :H, :W].contiguous()
#         else:
#             x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        
#         return x

# class SpecMambaBlock(nn.Module):
#     def __init__(
#         self,
#         dim,
#         num_blocks=2,
#     ):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             MaskedMambaVisionLayer(
#                 dim=dim,
#                 mixer_type="mamba" if i % 2 == 0 else "frequency",  # Alternate between frequency and mamba mixers
#                 mlp_ratio=4.,
#                 drop=0.,
#                 drop_path=0.,
#                 norm_layer=nn.LayerNorm,
#                 Mlp=Mlp,
#                 layer_scale=None,
#                 downsample=False,
#                 window_size=8,
#                 stage_idx=i,
#                 num_blocks=num_blocks,
#             )
#             for i in range(2*num_blocks) 
#         ])
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x
    

# class ConvBlock(nn.Module):

#     def __init__(self, dim,
#                  drop_path=0.,
#                  layer_scale=None,
#                  kernel_size=3):
#         super().__init__()

#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
#         self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
#         self.act1 = nn.GELU(approximate= 'tanh')
#         self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
#         self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
#         self.layer_scale = layer_scale
#         if layer_scale is not None and type(layer_scale) in [int, float]:
#             self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
#             self.layer_scale = True
#         else:
#             self.layer_scale = False
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         input = x
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.act1(x)
#         x = self.conv2(x)
#         x = self.norm2(x)
#         if self.layer_scale:
#             x = x * self.gamma.view(1, -1, 1, 1)
#         x = input + self.drop_path(x)
#         return x
    
# class Downsample(nn.Module):
#     """
#     Down-sampling block"
#     """

#     def __init__(self,
#                  dim,
#                  keep_dim=False,
#                  ):
#         """
#         Args:
#             dim: feature size dimension.
#             norm_layer: normalization layer.
#             keep_dim: bool argument for maintaining the resolution.
#         """

#         super().__init__()
#         if keep_dim:
#             dim_out = dim
#         else:
#             dim_out = 2 * dim
#         self.reduction = nn.Sequential(
#             nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
#         )

#     def forward(self, x):
#         x = self.reduction(x)
#         return x

# class Upsample(nn.Module):
#     """
#     Up-sampling block"
#     """

#     def __init__(self,
#                  dim,
#                  keep_dim=False,
#                  ):
#         """
#         Args:
#             dim: feature size dimension.
#             norm_layer: normalization layer.
#             keep_dim: bool argument for maintaining the resolution.
#         """

#         super().__init__()
#         if keep_dim:
#             dim_out = dim
#         else:
#             dim_out = dim // 2
#         self.up = nn.Sequential(
#             nn.ConvTranspose2d(dim, dim_out, 3, 2, 1, output_padding=1, bias=False),
#         )

#     def forward(self, x):
#         x = self.up(x)
#         return x

# class AttentionBlock(nn.Module):
#     """Attention block with learnable parameters"""

#     def __init__(self, F_g, F_l, n_coefficients):
#         """
#         :param F_g: number of feature maps (channels) in previous layer
#         :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
#         :param n_coefficients: number of learnable multi-dimensional attention coefficients
#         """
#         super(AttentionBlock, self).__init__()

#         self.W_gate = nn.Sequential(
#             nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(n_coefficients)
#         )

#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(n_coefficients)
#         )

#         self.psi = nn.Sequential(
#             nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, gate, skip_connection):
#         """
#         :param gate: gating signal from previous layer
#         :param skip_connection: activation from corresponding encoder layer
#         :return: output activations
#         """
#         g1 = self.W_gate(gate)
#         x1 = self.W_x(skip_connection)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         out = skip_connection * psi
#         return out

# class Specmambav2(nn.Module):
#     def __init__(self, n_channels=3, n_classes=2, dim=64, num_blocks=[2,2,2,2,2]):
#         super().__init__()

#         self.n_channels = n_channels
#         self.n_classes = n_classes

#         self.patch_embed = PatchEmbed(in_chans=n_channels, in_dim=dim, dim=dim)

#         self.conv_block_1 = nn.ModuleList([ConvBlock(dim=dim) for _ in range(num_blocks[0])]) # stage 1
#         self.downsample_1 = Downsample(dim=dim) #stage 1

#         self.conv_block_2 = nn.ModuleList([ConvBlock(dim=2*dim) for _ in range(num_blocks[1])]) # stage 2
#         self.downsample_2 = Downsample(dim=2*dim) #stage 2

#         self.spec_mamba_block_3 = SpecMambaBlock(dim=4*dim, num_blocks=num_blocks[2]) # stage 3
#         self.downsample_3 = Downsample(dim=4*dim) #stage 3

#         self.spec_mamba_block_4 = SpecMambaBlock(dim=8*dim, num_blocks=num_blocks[3]) # stage 4
#         self.downsample_4 = Downsample(dim=8*dim) #stage 4

#         self.upsample_4 = Upsample(dim=16*dim)
#         self.bottleneck_block = SpecMambaBlock(dim=16*dim, num_blocks=num_blocks[4]) # bottleneck

#         self.upsample_3 = Upsample(dim=8*dim) # stage 3
#         self.spec_mamba_block_up3 = SpecMambaBlock(dim=8*dim, num_blocks=num_blocks[3]) 

#         self.upsample_2 = Upsample(dim=4*dim) # stage 2
#         self.spec_mamba_block_up2 = SpecMambaBlock(dim=4*dim, num_blocks=num_blocks[2]) 

#         self.upsample_1 = Upsample(dim=2*dim, keep_dim=True) # stage 1    
#         self.spec_mamba_block_up1 = SpecMambaBlock(dim=2*dim, num_blocks=num_blocks[1]) 

#         self.att1 = AttentionBlock(F_g=2*dim, F_l=2*dim, n_coefficients=dim) # attention for stage 1
#         self.att2 = AttentionBlock(F_g=4*dim, F_l=4*dim, n_coefficients=2*dim) # attention for stage 2
#         self.att3 = AttentionBlock(F_g=8*dim, F_l=8*dim, n_coefficients=4*dim) # attention for stage 3 

#         # Final Reconstruction Layer
#         # To get back to H, W from H/4, W/4, we use a PixelShuffle or extra TransposeConvs
#         self.final_up = nn.Sequential(
#             nn.ConvTranspose2d(2*dim, dim, 2, 2), # H/2
#             nn.GELU(),
#             nn.ConvTranspose2d(dim, dim, 2, 2),   # H
#         )
        
#         self.output = nn.Conv2d(dim, n_classes, kernel_size=1, bias=False)

#     def forward(self, x):
#         # Initial Embedding
#         x = self.patch_embed(x) 
        
#         # Encoder
#         for blk in self.conv_block_1: x = blk(x)
#         skip1 = x 
#         x = self.downsample_1(x)

#         for blk in self.conv_block_2: x = blk(x)
#         skip2 = x
#         x = self.downsample_2(x)

#         x = self.spec_mamba_block_3(x)
#         skip3 = x
#         x = self.downsample_3(x)

#         x = self.spec_mamba_block_4(x)
#         skip4 = x
#         x = self.downsample_4(x)

#         # Bottleneck
#         x = self.bottleneck_block(x)

#         # Decoder
#         x = self.upsample_4(x)
#         x = self.att3(gate=x, skip_connection=skip4)
#         x = self.spec_mamba_block_up3(x)

#         x = self.upsample_3(x)
#         x = self.att2(gate=x, skip_connection=skip3)
#         x = self.spec_mamba_block_up2(x)

#         x = self.upsample_2(x)
#         x = self.att1(gate=x, skip_connection=skip2)
#         x = self.spec_mamba_block_up1(x)
#         x = self.upsample_1(x) # Final upsample to original resolution

#         # Return to full resolution
#         x = self.final_up(x)
#         x = self.output(x)

#         return x
    
# if __name__ == "__main__":
#     import torch
#     from thop import profile, clever_format
    
#     # Create a dummy input tensor (B, C, H, W)
#     # dummy input in the featreure space after the encoder, e.g., (B, 512, 16, 16) for a typical UNet bottleneck
#     t = torch.randn(1, 3, 512, 512).cuda()  # Example input shape
    
#     # Initialize the MaskedMambaVisionLayer
#     with torch.no_grad():
#         model =Specmambav2(n_channels=3, n_classes=2, dim=24, num_blocks=[2,2,4,4,2]).cuda()
#         y = model(t)
#         print(f"Output shape: {y.shape}")
    
#     # Profile with thop
#     print("\nProfiling with thop...")
#     macs, params = profile(model, inputs=(t,))
#     macs, params = clever_format([macs, params], "%.3f")
#     print(f"MACs: {macs}, Params: {params}")



#!/usr/bin/env python3
"""
Boundary-Normal SpecMamba MaskNet for SHIQ specular-highlight mask detection.

Main contribution:
    Boundary-Normal Specular-Aware Directional Scan (BN-SADS)

This model keeps SpecMambaBlock as the core contribution, but makes the scan
specular-geometry-aware:
    1. Predict a soft specular prior M from feature map F.
    2. Compute boundary normal n = grad(M) / ||grad(M)||.
    3. Run four directional selective scans:
        left -> right, right -> left, top -> bottom, bottom -> top.
    4. Bias directional fusion with boundary-normal alignment.
    5. Modulate selective-scan dt with specular probability, boundary strength,
       and direction alignment.

Use case:
    Lightweight specular-mask detection on SHIQ, e.g. 200x200 input.

Input convention:
    x: RGB image normalized to [-1, 1], shape (B, 3, H, W)

Output:
    raw logits, shape (B, 1, H, W)

Training:
    Use SpecularMaskLoss below. Do NOT sigmoid before loss.

Inference:
    prob = torch.sigmoid(logits)
    mask = (prob > threshold).float()
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

try:
    from timm.models.layers import DropPath, LayerNorm2d
except Exception:
    from timm.layers import DropPath, LayerNorm2d

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_map(x: Tensor, eps: float = 1e-6) -> Tensor:
    b = x.shape[0]
    flat = x.flatten(1)
    x_min = flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    x_max = flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    return (x - x_min) / (x_max - x_min + eps)


def sobel_xy(x: Tensor) -> Tuple[Tensor, Tensor]:
    kx = torch.tensor(
        [[[[-1.0, 0.0, 1.0],
           [-2.0, 0.0, 2.0],
           [-1.0, 0.0, 1.0]]]],
        device=x.device,
        dtype=x.dtype,
    )
    ky = torch.tensor(
        [[[[-1.0, -2.0, -1.0],
           [0.0, 0.0, 0.0],
           [1.0, 2.0, 1.0]]]],
        device=x.device,
        dtype=x.dtype,
    )
    x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(x_pad, kx)
    gy = F.conv2d(x_pad, ky)
    return gx, gy


# -----------------------------------------------------------------------------
# Prompt modules
# -----------------------------------------------------------------------------

class SpecularVisualPrior(nn.Module):
    """
    Builds low-level specular prompt maps from RGB.

    Input:
        x: [B, 3, H, W]
           If input_range='01', x is assumed in [0, 1].
           If input_range='minus1_1', x is assumed in [-1, 1].

    Output prior channels:
        brightness      : max RGB
        # saturation      : max RGB - min RGB
        achromaticity   : 1 - saturation / (brightness + 1e-6)
        white_likeness  : brightness * (1 - saturation)
        local_contrast  : normalized Sobel magnitude of luminance
    """
    def __init__(self, input_range: str = "01"):
        super().__init__()
        if input_range not in {"01", "minus1_1"}:
            raise ValueError("input_range must be '01' or 'minus1_1'.")
        self.input_range = input_range

    def _to_01(self, x: Tensor) -> Tensor:
        if self.input_range == "01":
            return x.clamp(0.0, 1.0)
        return (x * 0.5 + 0.5).clamp(0.0, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        x01 = self._to_01(x)
        rgb_max = x01.amax(dim=1, keepdim=True)
        rgb_min = x01.amin(dim=1, keepdim=True)

        brightness = rgb_max
        # saturation = rgb_max - rgb_min
        achromaticity = 1.0 - (rgb_max - rgb_min) / (rgb_max + 1e-6)
        saturation = (rgb_max - rgb_min) / (rgb_max + 1e-6)
        white_likeness = brightness * (1.0 - saturation)

        gray = 0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]
        gx, gy = sobel_xy(gray)
        local_contrast = torch.sqrt(gx * gx + gy * gy + 1e-6)
        local_contrast = normalize_map(local_contrast)

        return torch.cat([brightness, achromaticity, white_likeness, local_contrast], dim=1)

def reflectance_prompt_pseudo_target(prior: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Build weak pseudo prompt targets from the 4-channel specular visual prior.

    prior: [B, 4, H, W]
        channel 0: brightness
        channel 1: achromaticity
        channel 2: white_likeness
        channel 3: local_contrast

    returns:
        q: [B, 6]

    Prompt meaning:
        0: compact saturated/strong highlight
        1: large glare
        2: elongated / structured reflection
        3: weak / soft specular highlight
        4: boundary / halo
        5: bright diffuse non-specular distractor
    """
    assert prior.dim() == 4, "prior must be [B, 4, H, W]"
    assert prior.shape[1] == 4, "Expected 4 prior channels."

    b, c, h, w = prior.shape

    brightness = prior[:, 0].flatten(1)      # [B, HW]
    achroma = prior[:, 1].flatten(1)         # [B, HW]
    white = prior[:, 2].flatten(1)           # [B, HW]
    contrast = prior[:, 3].flatten(1)        # [B, HW]

    mean_b = brightness.mean(dim=1)
    mean_a = achroma.mean(dim=1)
    mean_w = white.mean(dim=1)
    mean_c = contrast.mean(dim=1)

    area_bright = (brightness > 0.75).float().mean(dim=1)
    area_white = (white > 0.65).float().mean(dim=1)
    area_achroma = (achroma > 0.70).float().mean(dim=1)
    area_contrast = (contrast > 0.50).float().mean(dim=1)

    k_small = max(1, int(0.03 * h * w))
    k_large = max(1, int(0.10 * h * w))

    top_b_small = brightness.topk(k_small, dim=1).values.mean(dim=1)
    top_b_large = brightness.topk(k_large, dim=1).values.mean(dim=1)

    top_w_small = white.topk(k_small, dim=1).values.mean(dim=1)
    top_w_large = white.topk(k_large, dim=1).values.mean(dim=1)

    # 0: compact strong highlight
    # Bright, white/achromatic, spatially sparse.
    p0 = top_b_small * top_w_small * mean_a * (1.0 - area_bright)

    # 1: large glare
    # Bright and achromatic over a large area.
    p1 = top_b_large * top_w_large * area_bright * area_achroma

    # 2: elongated / structured reflection
    # Your current prior has no explicit orientation cue, so this is approximate.
    # Structured reflection: bright + high contrast + medium spatial support.
    p2 = mean_b * mean_c * area_contrast * area_bright * (1.0 - area_bright)

    # 3: weak / soft specular highlight
    # Moderately bright, achromatic, low boundary contrast, not too large.
    p3 = mean_b * mean_a * (1.0 - mean_c) * (1.0 - area_bright)

    # 4: boundary / halo
    # White/achromatic region with high local contrast.
    p4 = mean_w * mean_c * area_white

    # 5: bright diffuse non-specular distractor
    # Bright and achromatic/white, but low contrast.
    # This separates smooth white objects from true specular boundaries.
    p5 = mean_b * mean_w * mean_a * (1.0 - mean_c)

    q = torch.stack([p0, p1, p2, p3, p4, p5], dim=1)
    q = q.clamp_min(eps)
    q = q / (q.sum(dim=1, keepdim=True) + eps)

    return q

def prompt_diversity_loss(
    weights: Tensor,
    entropy_weight: float = 0.01,
    balance_weight: float = 0.05,
    eps: float = 1e-6,
) -> Tensor:
    """
    weights: [B, K]

    Prevents prompt collapse.
    """
    # Encourage each image not to collapse too early to a single prompt.
    entropy = -(weights * (weights + eps).log()).sum(dim=1).mean()
    entropy_loss = -entropy

    # Encourage batch-level prompt usage to be balanced.
    avg_w = weights.mean(dim=0)
    uniform = torch.full_like(avg_w, 1.0 / avg_w.numel())

    balance_loss = F.kl_div(
        (avg_w + eps).log(),
        uniform,
        reduction="batchmean",
    )

    return entropy_weight * entropy_loss + balance_weight * balance_loss

def prompt_routing_loss(
    weights: Tensor,
    pseudo_q: Tensor,
    route_weight: float = 0.05,
    eps: float = 1e-6,
) -> Tensor:
    """
    weights:  [B, 6] predicted prompt distribution
    pseudo_q: [B, 6] weak pseudo target distribution
    """
    loss = F.kl_div(
        (weights + eps).log(),
        pseudo_q.detach(),
        reduction="batchmean",
    )
    return route_weight * loss

def reflectance_prompt_loss(
    weights: Tensor,
    prior: Tensor,
    entropy_weight: float = 0.01,
    balance_weight: float = 0.05,
    route_weight: float = 0.05,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    weights: [B, 6]
    prior:   [B, 4, H, W]
    """
    pseudo_q = reflectance_prompt_pseudo_target(prior)

    div_loss = prompt_diversity_loss(
        weights,
        entropy_weight=entropy_weight,
        balance_weight=balance_weight,
    )

    route_loss = prompt_routing_loss(
        weights,
        pseudo_q,
        route_weight=route_weight,
    )

    total = div_loss + route_loss

    logs = {
        "prompt_div_loss": div_loss.detach(),
        "prompt_route_loss": route_loss.detach(),
        "prompt_total_loss": total.detach(),
        "prompt_entropy": (-(weights * (weights + 1e-6).log()).sum(dim=1).mean()).detach(),
        "prompt_usage": weights.mean(dim=0).detach(),
    }

    return total, logs

class ReflectancePromptBank(nn.Module):
    """
    Learnable reflectance prompt bank.

    The prompts are not text tokens; they are learnable low-level reflectance states:
        0: small saturated highlight
        1: large glare
        2: elongated reflection
        3: weak/soft specular highlight
        4: boundary/halo
        5: bright diffuse non-specular region
    """
    def __init__(self, prompt_dim: int = 128, num_prompts: int = 6, prior_channels: int = 4):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts

        self.prompt_tokens = nn.Parameter(torch.randn(num_prompts, prompt_dim) * 0.02)
        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(prior_channels, prompt_dim, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(prompt_dim, num_prompts, 1, bias=True),
        )

    def forward(self, prior: Tensor) -> Tuple[Tensor, Tensor]:
        b = prior.shape[0]
        logits = self.selector(prior).view(b, self.num_prompts)
        weights = torch.softmax(logits, dim=1)
        prompt_embed = weights @ self.prompt_tokens
        return prompt_embed, weights


class PromptFiLM2d(nn.Module):
    """Prompt-conditioned channel modulation."""
    def __init__(self, channels: int, prompt_dim: int = 128, scale: float = 0.10):
        super().__init__()
        self.scale = scale
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(prompt_dim, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels * 2),
        )
        nn.init.zeros_(self.to_gamma_beta[-1].weight)
        nn.init.zeros_(self.to_gamma_beta[-1].bias)

    def forward(self, feat: Tensor, prompt_embed: Tensor) -> Tensor:
        gamma, beta = self.to_gamma_beta(prompt_embed).chunk(2, dim=1)
        gamma = self.scale * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = self.scale * torch.tanh(beta).unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + gamma) + beta


class PromptSpatialGate(nn.Module):
    """
    Spatial prompt gate from feature map and specular prior.

    This is intended to suppress bright diffuse false positives while enhancing
    likely specular regions.
    """
    def __init__(self, channels: int, prior_channels: int = 4):
        super().__init__()
        groups = 8
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.gate = nn.Sequential(
            nn.Conv2d(channels + prior_channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feat: Tensor, prior: Tensor) -> Tuple[Tensor, Tensor]:
        if prior.shape[-2:] != feat.shape[-2:]:
            prior = F.interpolate(prior, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.gate(torch.cat([feat, prior], dim=1))
        return feat * (1.0 + gate), gate


# -----------------------------------------------------------------------------
# Lightweight convolution blocks
# -----------------------------------------------------------------------------

class GNAct(nn.Module):
    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        groups = min(num_groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(x))


class DSConvBlock(nn.Module):
    def __init__(self, channels: int, expansion: float = 1.0):
        super().__init__()
        hidden = int(channels * expansion)
        self.pw1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.n1 = GNAct(hidden)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.n2 = GNAct(hidden)
        self.pw2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.norm_out = nn.GroupNorm(1, channels)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-2)

    def forward(self, x: Tensor) -> Tensor:
        y = self.pw1(x)
        y = self.n1(y)
        y = self.dw(y)
        y = self.n2(y)
        y = self.pw2(y)
        y = self.norm_out(y)
        return x + self.gamma * y


class ECA(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class LiteStage(nn.Module):
    def __init__(self, channels: int, depth: int, expansion: float = 1.0):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [DSConvBlock(channels, expansion=expansion), ECA(channels)]
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            GNAct(out_ch),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down(x)


class UpFuse(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        gate_ch = max(skip_ch // 4, 8)
        self.skip_gate = nn.Sequential(
            nn.Conv2d(skip_ch, gate_ch, 1, bias=False),
            GNAct(gate_ch),
            nn.Conv2d(gate_ch, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False),
            GNAct(out_ch),
            DSConvBlock(out_ch),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = skip * self.skip_gate(skip)
        return self.fuse(torch.cat([x, skip], dim=1))


# -----------------------------------------------------------------------------
# Specular prior and boundary-normal geometry
# -----------------------------------------------------------------------------

class SpecPriorPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32):
        super().__init__()
        hidden = max(16, min(hidden, in_channels))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            LayerNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            LayerNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BoundaryNormalEstimator(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, M: Tensor) -> Tuple[Tensor, Tensor]:
        gx, gy = sobel_xy(M)
        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + self.eps)
        nx = gx / (mag + self.eps)
        ny = gy / (mag + self.eps)
        align = torch.cat([F.relu(nx), F.relu(-nx), F.relu(ny), F.relu(-ny)], dim=1)
        boundary = normalize_map(mag)
        return boundary, align


# -----------------------------------------------------------------------------
# Selective scan core
# -----------------------------------------------------------------------------

class SelectiveScan1DMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 1,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        if self.d_inner % 2 != 0:
            raise ValueError("d_inner must be even because it is split into x and z.")
        self.d_half = self.d_inner // 2
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(self.d_half, self.dt_rank + 2 * d_state, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_half, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unknown dt_init={dt_init}")

        dt = torch.exp(
            torch.rand(self.d_half, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=self.d_half).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_half, device=device))
        self.D._no_weight_decay = True

        self.conv1d_x = nn.Conv1d(
            self.d_half,
            self.d_half,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.d_half,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            self.d_half,
            self.d_half,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.d_half,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)

        self.alpha_mask = nn.Parameter(torch.tensor(0.15))
        self.beta_boundary = nn.Parameter(torch.tensor(0.20))

    def forward(self, tokens: Tensor, mask_seq: Optional[Tensor] = None, gate_seq: Optional[Tensor] = None) -> Tensor:
        b, seqlen, _ = tokens.shape

        xz = self.in_proj(tokens)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        x = F.silu(self.conv1d_x(x))
        z = F.silu(self.conv1d_z(z))
        if x.shape[-1] != seqlen:
            x = x[..., :seqlen]
            z = z[..., :seqlen]

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B_scan, C_scan = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = self.dt_proj(dt)
        dt = rearrange(dt, "(b l) d -> b d l", b=b, l=seqlen)
        dt = F.softplus(dt)

        # Important for AMP:
        # mask_seq / gate_seq may be fp32, while x is fp16 under autocast.
        # Cast them to dt dtype before multiplication to avoid promoting dt to fp32.
        if mask_seq is not None:
            mask_seq = mask_seq.to(device=dt.device, dtype=dt.dtype)
            alpha = self.alpha_mask.sigmoid().to(dtype=dt.dtype)
            dt = dt * (1.0 + alpha * mask_seq)

        if gate_seq is not None:
            gate_seq = gate_seq.to(device=dt.device, dtype=dt.dtype)
            beta = self.beta_boundary.sigmoid().to(dtype=dt.dtype)
            dt = dt * (1.0 + beta * gate_seq)

        dt = dt.clamp(min=1e-4, max=0.2)

        # Mamba selective_scan CUDA requires u and delta to have the same dtype.
        scan_dtype = x.dtype
        x = x.contiguous()
        z = z.to(dtype=scan_dtype).contiguous()
        dt = dt.to(dtype=scan_dtype).contiguous()

        B_scan = rearrange(B_scan, "(b l) n -> b n l", b=b, l=seqlen)
        C_scan = rearrange(C_scan, "(b l) n -> b n l", b=b, l=seqlen)

        B_scan = B_scan.to(dtype=scan_dtype).contiguous()
        C_scan = C_scan.to(dtype=scan_dtype).contiguous()

        # A should stay fp32 for numerical stability.
        A = -torch.exp(self.A_log.float())

        y = selective_scan_fn(
            x,
            dt,
            A,
            B_scan,
            C_scan,
            self.D.float(),
            z=None,
            delta_bias=None,
            delta_softplus=False,
            return_last_state=None,
        )

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)


# -----------------------------------------------------------------------------
# Boundary-Normal Specular-Aware Directional Scan
# -----------------------------------------------------------------------------

class LocalHighFreqBranch(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.low = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            LayerNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        low = self.low(x)
        high = x - low
        return self.proj(torch.cat([low, high], dim=1))


class BoundaryNormalDirectionalScan(nn.Module):
    def __init__(self, dim: int, d_state: int = 8, d_conv: int = 3, expand: int = 1, shared_scan: bool = True):
        super().__init__()
        self.dim = dim
        self.shared_scan = shared_scan
        self.spec_prior = SpecPriorPredictor(dim)
        self.normal_estimator = BoundaryNormalEstimator()

        if shared_scan:
            self.scan = SelectiveScan1DMixer(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.scan_lr = SelectiveScan1DMixer(dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self.scan_rl = SelectiveScan1DMixer(dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self.scan_tb = SelectiveScan1DMixer(dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self.scan_bt = SelectiveScan1DMixer(dim, d_state=d_state, d_conv=d_conv, expand=expand)

        self.local_branch = LocalHighFreqBranch(dim)

        gate_hidden = max(dim // 2, 8)
        self.learned_gate = nn.Sequential(
            nn.Conv2d(dim, gate_hidden, 1, bias=False),
            LayerNorm2d(gate_hidden),
            nn.GELU(),
            nn.Conv2d(gate_hidden, 5, 1, bias=True),
        )
        self.boundary_gate_strength = nn.Parameter(torch.tensor(1.0))
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            LayerNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
        )

    def _get_scan(self, name: str) -> SelectiveScan1DMixer:
        if self.shared_scan:
            return self.scan
        return getattr(self, f"scan_{name}")

    def _scan_lr(self, x: Tensor, M: Tensor, align: Tensor) -> Tensor:
        b, c, h, w = x.shape
        tokens = rearrange(x, "b c h w -> (b h) w c")
        mask_seq = rearrange(M, "b one h w -> (b h) one w")
        align_seq = rearrange(align, "b one h w -> (b h) one w")
        y = self._get_scan("lr")(tokens, mask_seq, align_seq)
        return rearrange(y, "(b h) w c -> b c h w", b=b, h=h, w=w)

    def _scan_rl(self, x: Tensor, M: Tensor, align: Tensor) -> Tensor:
        x_f = torch.flip(x, dims=[-1])
        M_f = torch.flip(M, dims=[-1])
        a_f = torch.flip(align, dims=[-1])
        y = self._scan_lr(x_f, M_f, a_f)
        return torch.flip(y, dims=[-1])

    def _scan_tb(self, x: Tensor, M: Tensor, align: Tensor) -> Tensor:
        b, c, h, w = x.shape
        tokens = rearrange(x, "b c h w -> (b w) h c")
        mask_seq = rearrange(M, "b one h w -> (b w) one h")
        align_seq = rearrange(align, "b one h w -> (b w) one h")
        y = self._get_scan("tb")(tokens, mask_seq, align_seq)
        return rearrange(y, "(b w) h c -> b c h w", b=b, h=h, w=w)

    def _scan_bt(self, x: Tensor, M: Tensor, align: Tensor) -> Tensor:
        x_f = torch.flip(x, dims=[-2])
        M_f = torch.flip(M, dims=[-2])
        a_f = torch.flip(align, dims=[-2])
        y = self._scan_tb(x_f, M_f, a_f)
        return torch.flip(y, dims=[-2])

    def forward(self, x: Tensor, return_aux: bool = False):
        M = self.spec_prior(x)
        boundary, align4 = self.normal_estimator(M)

        y_lr = self._scan_lr(x, M, boundary * align4[:, 0:1])
        y_rl = self._scan_rl(x, M, boundary * align4[:, 1:2])
        y_tb = self._scan_tb(x, M, boundary * align4[:, 2:3])
        y_bt = self._scan_bt(x, M, boundary * align4[:, 3:4])
        y_local = self.local_branch(x)

        branches = torch.stack([y_lr, y_rl, y_tb, y_bt, y_local], dim=1)
        learned = self.learned_gate(x)
        normal_bias5 = torch.cat([boundary * align4, torch.zeros_like(boundary)], dim=1)
        gate_logits = learned + self.boundary_gate_strength * normal_bias5
        gate = torch.softmax(gate_logits, dim=1).unsqueeze(2)

        y = (branches * gate).sum(dim=1)
        y = self.out_proj(y)

        if return_aux:
            return y, {
                "spec_prior_scan": M,
                "boundary": boundary,
                "align4": align4,
                "scan_gate": gate.squeeze(2),
            }
        return y


class ConvFFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            LayerNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            LayerNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BoundaryNormalSpecMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 1,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,
        shared_scan: bool = True,
        layer_scale: float = 1e-2,
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.mixer = BoundaryNormalDirectionalScan(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            shared_scan=shared_scan,
        )
        self.norm2 = LayerNorm2d(dim)
        self.ffn = ConvFFN(dim, mlp_ratio=mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        y = self.mixer(self.norm1(x))
        x = x + self.drop_path(self.gamma_1.view(1, -1, 1, 1) * y)
        y = self.ffn(self.norm2(x))
        x = x + self.drop_path(self.gamma_2.view(1, -1, 1, 1) * y)
        return x


class SpecMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 1,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 1,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,
        shared_scan: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            BoundaryNormalSpecMambaBlock(
                dim=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
                shared_scan=shared_scan,
            )
            for _ in range(depth)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# -----------------------------------------------------------------------------
# Mixture-of-Experts dual head (detection + removal)
# -----------------------------------------------------------------------------

class ReflectanceRegimeMoE(nn.Module):
    """R2-MoE: a single reflectance-regime router shared by detection and removal.

    Core idea (the contribution): specular detection and removal are the SAME
    physical decision read out two ways. A single per-pixel router assigns each
    location to one of K reflectance regimes (diffuse / soft-highlight /
    saturated-core / boundary). Both tasks consume the SAME routing:

        detection :  mask_logits = readout(routing)         -- which regimes are specular
        removal   :  residual    = sum_k routing_k * E_k(x) -- regime-mixed correction

    The experts have INCREASING receptive field (dilation 1,2,4,...), so the
    router can send easy local regimes (diffuse) to small-context experts and
    hard regimes (saturated cores, which must be inpainted from context) to
    large-context experts. The router is supervised by both losses, coupling
    detection and removal through one physically-interpretable routing.

    Returns:
        mask_logits : (B, n_classes, H, W)
        residual    : (B, out_channels, H, W)  -- added to the input image
        gate        : (B, K, H, W) softmax regime routing (interpretable + for balancing)
    """

    def __init__(self, dim: int, n_experts: int = 4, n_classes: int = 2, out_channels: int = 3):
        super().__init__()
        self.n_experts = n_experts
        # Regime experts with increasing receptive field (context grows with regime difficulty).
        self.experts = nn.ModuleList()
        for k in range(n_experts):
            d = 2 ** k  # dilation: 1, 2, 4, 8, ... -> more context for harder regimes
            self.experts.append(nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=d, dilation=d, groups=dim, bias=False),
                nn.Conv2d(dim, dim, 1, bias=False),
                LayerNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, out_channels, 1, bias=True),   # each expert emits a residual contribution
            ))
        # Shared reflectance-regime router (one routing for both tasks).
        self.router = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            LayerNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, n_experts, 1, bias=True),
        )
        # Detection is a learned read-out of the routing (which regimes count as specular).
        self.mask_readout = nn.Conv2d(n_experts, n_classes, 1, bias=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        gate = torch.softmax(self.router(x), dim=1)                    # (B,K,H,W) regime routing
        residual = None
        for k, expert in enumerate(self.experts):
            contrib = gate[:, k:k + 1] * expert(x)                     # (B,out,H,W)
            residual = contrib if residual is None else residual + contrib
        mask_logits = self.mask_readout(gate)                          # detection FROM the routing
        return mask_logits, residual, gate


class RemovalResBlock(nn.Module):
    """Residual block whose features are FiLM-modulated by the regime gate.
    The (scale, shift) come from the R2-MoE routing, so removal refinement is
    conditioned per-pixel on the predicted reflectance regime."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.pw1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, f: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
        h = self.norm(f)
        h = h * (1.0 + scale) + shift            # regime-gated FiLM
        h = self.pw2(self.act(self.pw1(self.dw(h))))
        return f + h


class RegimeGatedRemovalMoE(nn.Module):
    """R2-MoE with a dedicated, full-resolution, regime-gated removal decoder.

    Same shared router as ReflectanceRegimeMoE (one routing drives BOTH detection
    and removal -- the contribution is intact). The difference: removal is no
    longer a thin per-expert pixel read-out. Instead:
      - regime experts (increasing receptive field) emit FEATURES,
      - the gate FiLM-modulates a stack of refinement blocks (real depth),
      - features are upsampled to FULL resolution via PixelShuffle, then refined,
      - a final conv emits the residual at full res (no bilinear residual upsample).
    This gives removal genuine reconstruction capacity + high-frequency detail,
    which is the PSNR bottleneck, while the regime routing still guides everything.
    """
    def __init__(self, dim: int, n_experts: int = 4, n_classes: int = 2,
                 out_channels: int = 3, removal_blocks: int = 4, upscale: int = 1,
                 removal_dim: Optional[int] = None):
        super().__init__()
        self.n_experts = n_experts
        self.upscale = upscale
        # Removal is reconstruction-heavy: run the feature/reconstruction pathway at a
        # wider `rdim` while the router/gate stay tied to the input `dim` (the routing
        # is shared with detection, so its width must match the backbone feature `dim`).
        rdim = int(removal_dim) if removal_dim else dim
        self.rdim = rdim
        self.experts = nn.ModuleList()
        for k in range(n_experts):
            d = 2 ** k
            # depthwise context at `dim`, then project up to `rdim`.
            self.experts.append(nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=d, dilation=d, groups=dim, bias=False),
                nn.Conv2d(dim, rdim, 1, bias=False),
                LayerNorm2d(rdim), nn.GELU(),
            ))
        self.router = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            LayerNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, n_experts, 1, bias=True),
        )
        self.mask_readout = nn.Conv2d(n_experts, n_classes, 1, bias=True)

        # fuse raw backbone feature (dim) with regime-mixed feature (rdim) -> rdim.
        self.fuse = nn.Sequential(
            nn.Conv2d(dim + rdim, rdim, 1, bias=False), LayerNorm2d(rdim), nn.GELU(),
        )
        self.gate_film = nn.Conv2d(n_experts, rdim * 2, 1, bias=True)
        self.blocks = nn.ModuleList([RemovalResBlock(rdim) for _ in range(removal_blocks)])
        if upscale > 1:
            self.up = nn.Sequential(
                nn.Conv2d(rdim, rdim * upscale * upscale, 3, padding=1, bias=False),
                nn.PixelShuffle(upscale), LayerNorm2d(rdim), nn.GELU(),
            )
        else:
            self.up = nn.Identity()
        self.refine = nn.Sequential(
            nn.Conv2d(rdim, rdim, 3, padding=1, bias=False), LayerNorm2d(rdim), nn.GELU(),
        )
        self.to_residual = nn.Conv2d(rdim, out_channels, 3, padding=1, bias=True)
        nn.init.zeros_(self.to_residual.weight)            # start as identity (D_hat = x)
        nn.init.zeros_(self.to_residual.bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        gate = torch.softmax(self.router(x), dim=1)         # (B,K,h,w) regime routing
        mix = None
        for k, expert in enumerate(self.experts):
            contrib = gate[:, k:k + 1] * expert(x)          # (B,rdim,h,w)
            mix = contrib if mix is None else mix + contrib
        mask_logits = self.mask_readout(gate)               # detection FROM routing

        f = self.fuse(torch.cat([x, mix], dim=1))           # raw(dim) + regime-mixed(rdim) -> rdim
        scale, shift = self.gate_film(gate).chunk(2, dim=1)
        for blk in self.blocks:
            f = blk(f, scale, shift)                         # gate-conditioned refinement
        f = self.up(f)                                      # -> full resolution (PixelShuffle)
        f = self.refine(f)
        residual = self.to_residual(f)                      # full-res residual
        return mask_logits, residual, gate


def moe_balance_loss(gate: Tensor) -> Tensor:
    """Load-balancing penalty so regime experts are used evenly (Switch-Transformer style).

    gate: (B, K, H, W) softmax routing. Penalises deviation of mean per-expert
    usage from uniform 1/K, preventing regime collapse.
    """
    K = gate.shape[1]
    importance = gate.mean(dim=(0, 2, 3))                  # (K,)
    return K * (importance - 1.0 / K).pow(2).sum()


def regime_prior_loss(gate: Tensor, spec_prob: Tensor, sat_mask: Tensor) -> Tensor:
    """Weak physical anchoring so regimes are interpretable (optional, lam_reg).

    Encourages regime 0 -> diffuse (non-specular) and the last regime ->
    saturated core. Targets are detached so the router, not the cues, adapts.
        gate     : (B, K, H, W) routing
        spec_prob: (B, 1, H, W) soft specular probability in [0, 1]
        sat_mask : (B, 1, H, W) saturated-region mask in [0, 1]
    """
    K = gate.shape[1]
    diffuse_target = (1.0 - spec_prob).detach()
    core_target = sat_mask.detach()
    loss_diffuse = F.l1_loss(gate[:, 0:1], diffuse_target)
    loss_core = F.l1_loss(gate[:, K - 1:K], core_target)
    return loss_diffuse + loss_core


# -----------------------------------------------------------------------------
# Prompted Specmambav3
# -----------------------------------------------------------------------------

class Specmamba(nn.Module):
    """
    Reflectance-prompted SpecMambaV2 for SHIQ specular-mask detection.

    Input:
        x: [B, 3, H, W], normally in [0, 1] from BasicDataset.

    Output:
        logits: [B, n_classes, H, W]

    Recommended first configuration:
        PromptedSpecMambav2(
            base_dim=32,
            depths=(2, 2, 3, 3),
            shared_scan=True,
            prompt_dim=128,
            input_range="01",
        )
    """
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 2,
        base_dim: int = 32,
        depths: Tuple[int, int, int, int] = (2, 2, 4, 4),
        shared_scan: bool = True,
        prompt_dim: int = 128,
        num_prompts: int = 6,
        input_range: str = "01",
        use_spatial_prompt_gate: bool = True,
        stem_stride: int = 1,
        lite_expansion: float = 1.0,
        use_mamba_decoder: bool = False,
        use_mamba_stage3: bool = True,
        dual_head: bool = False,
        moe_experts: int = 4,
        deep_removal: bool = False,
        removal_blocks: int = 4,
        removal_dim: Optional[int] = None,
    ):

        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.dual_head = dual_head
        self.use_spatial_prompt_gate = use_spatial_prompt_gate
        self.stem_stride = stem_stride

        c1 = base_dim
        c2 = base_dim * 2
        c3 = base_dim * 4
        c4 = base_dim * 8

        self.visual_prior = SpecularVisualPrior(input_range=input_range)
        self.prompt_bank = ReflectancePromptBank(
            prompt_dim=prompt_dim,
            num_prompts=num_prompts,
            prior_channels=4,
        )

        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, c1, 3, stride=stem_stride, padding=1, bias=False),
            GNAct(c1),
            nn.Conv2d(c1, c1, 3, padding=1, bias=False),
            GNAct(c1),
        )

        self.prompt_film1 = PromptFiLM2d(c1, prompt_dim=prompt_dim)
        self.prompt_film2 = PromptFiLM2d(c2, prompt_dim=prompt_dim)
        self.prompt_film3 = PromptFiLM2d(c3, prompt_dim=prompt_dim)
        self.prompt_film4 = PromptFiLM2d(c4, prompt_dim=prompt_dim)

        if use_spatial_prompt_gate:
            self.prompt_gate1 = PromptSpatialGate(c1, prior_channels=4)
            self.prompt_gate2 = PromptSpatialGate(c2, prior_channels=4)
            # self.prompt_gate3 = PromptSpatialGate(c3, prior_channels=4)
            self.prompt_gate3 = None
        else:
            self.prompt_gate1 = None
            self.prompt_gate2 = None
            self.prompt_gate3 = None

        # self.enc1 = LiteStage(c1, depths[0], expansion=lite_expansion)
        self.enc1 = SpecMambaBlock(c1, depth=depths[0], shared_scan=shared_scan)
        self.down1 = Down(c1, c2)
        # self.enc2 = LiteStage(c2, depths[1], expansion=lite_expansion)
        self.enc2 = SpecMambaBlock(c2, depth=depths[1], shared_scan=shared_scan)
        self.down2 = Down(c2, c3)

        self.enc3 = SpecMambaBlock(c3, depth=depths[2], shared_scan=shared_scan) if use_mamba_stage3 else LiteStage(c3, depths[2], expansion=lite_expansion)
        self.down3 = Down(c3, c4)
        self.bottleneck = SpecMambaBlock(c4, depth=depths[3], shared_scan=shared_scan)

        self.up3 = UpFuse(c4, c3, c3)
        self.dec3 = SpecMambaBlock(c3, depth=1, shared_scan=shared_scan) if use_mamba_decoder else LiteStage(c3, 1, expansion=lite_expansion)

        self.up2 = UpFuse(c3, c2, c2)
        self.dec2 = LiteStage(c2, 1, expansion=lite_expansion)

        self.up1 = UpFuse(c2, c1, c1)
        self.dec1 = LiteStage(c1, 1, expansion=lite_expansion)

        self.output = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, bias=False),
            GNAct(c1),
            nn.Conv2d(c1, n_classes, 1, bias=True),
        )

        # Unified detection + removal: shared reflectance-regime MoE router (R2-MoE).
        if dual_head:
            if deep_removal:
                # regime-gated full-res removal decoder (strong removal path)
                self.dual = RegimeGatedRemovalMoE(
                    c1, n_experts=moe_experts, n_classes=n_classes, out_channels=n_channels,
                    removal_blocks=removal_blocks, upscale=stem_stride, removal_dim=removal_dim)
            else:
                # lightweight per-expert pixel read-out (original)
                self.dual = ReflectanceRegimeMoE(c1, n_experts=moe_experts, n_classes=n_classes, out_channels=n_channels)

    @property
    def num_parameters(self) -> int:
        return count_parameters(self)

    def _apply_spatial_gate(self, feat: Tensor, prior: Tensor, gate_module: Optional[nn.Module]):
        if gate_module is None:
            return feat, None
        return gate_module(feat, prior)

    def forward(self, x: Tensor, return_aux: bool = False):
        input_size = x.shape[-2:]
        prior = self.visual_prior(x)
        prompt_embed, prompt_weight = self.prompt_bank(prior)

        x1 = self.stem(x)
        x1 = self.prompt_film1(x1, prompt_embed)
        x1, gate1 = self._apply_spatial_gate(x1, prior, self.prompt_gate1)
        x1 = self.enc1(x1)

        x2 = self.down1(x1)
        x2 = self.prompt_film2(x2, prompt_embed)
        x2, gate2 = self._apply_spatial_gate(x2, prior, self.prompt_gate2)
        x2 = self.enc2(x2)

        x3 = self.down2(x2)
        x3 = self.prompt_film3(x3, prompt_embed)
        x3, gate3 = self._apply_spatial_gate(x3, prior, self.prompt_gate3)
        x3 = self.enc3(x3)

        x4 = self.down3(x3)
        x4 = self.prompt_film4(x4, prompt_embed)
        x4 = self.bottleneck(x4)

        y3 = self.up3(x4, x3)
        y3 = self.prompt_film3(y3, prompt_embed)
        y3 = self.dec3(y3)

        y2 = self.up2(y3, x2)
        y2 = self.prompt_film2(y2, prompt_embed)
        y2 = self.dec2(y2)

        y1 = self.up1(y2, x1)
        y1 = self.prompt_film1(y1, prompt_embed)
        y1 = self.dec1(y1)

        # ---- unified detection + removal via the shared regime router (R2-MoE) ----
        if self.dual_head:
            mask_logits, residual, regime_gate = self.dual(y1)
            if mask_logits.shape[-2:] != input_size:
                mask_logits = F.interpolate(mask_logits, size=input_size, mode="bilinear", align_corners=False)
            # deep_removal returns a full-res residual; shallow head returns it at y1 res.
            if residual.shape[-2:] != input_size:
                residual = F.interpolate(residual, size=input_size, mode="bilinear", align_corners=False)
            # global-residual restoration: preserves input detail, learns the correction.
            D_hat = (x + residual).clamp(0.0, 1.0)
            if not return_aux:
                return mask_logits, D_hat
            aux: Dict[str, Tensor] = {
                "visual_prior": prior,
                "prompt_weight": prompt_weight,
                "regime_gate": regime_gate,
                "removal_residual": residual,
            }
            return mask_logits, D_hat, aux

        logits = self.output(y1)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        if not return_aux:
            return logits

        aux: Dict[str, Tensor] = {
            "visual_prior": prior,
            "prompt_weight": prompt_weight,
        }
        if gate1 is not None:
            aux["prompt_gate1"] = gate1
        if gate2 is not None:
            aux["prompt_gate2"] = gate2
        if gate3 is not None:
            aux["prompt_gate3"] = gate3
        return logits, aux


# # Backward-compatible aliases if you want to import this file like old models.
# SpecmambaPrompt = PromptedSpecMambav2
# Specmambav2 = PromptedSpecMambav2


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = PromptedSpecMambav2(
    #                             base_dim=40,
    #                             depths=(2, 2, 4, 4),
    #                             shared_scan=True,
    #                             prompt_dim=128,
    #                             input_range="01",
    #                             use_spatial_prompt_gate=True,
    #                         ).to(device)
    
    model = Specmamba(
                            base_dim=32,
                            depths=(2, 2, 4, 4),
                            shared_scan=True,
                            prompt_dim=128,
                            input_range="01",
                            stem_stride=2,
                            lite_expansion=1.0,
                            use_mamba_decoder=False,
                            use_mamba_stage3=True,
                            dual_head=True,
                            moe_experts=4,
                        ).cuda()

    x = torch.randn(1, 3, 512, 512, device=device).sigmoid()
    with torch.no_grad():
        mask_logits, D_hat, aux = model(x, return_aux=True)
    print(f"mask_logits: {tuple(mask_logits.shape)}  D_hat: {tuple(D_hat.shape)}")
    g = aux['regime_gate']
    print(f"regime_gate: {tuple(g.shape)}  (per-expert usage: {g.mean(dim=(0,2,3)).tolist()})")
    print(f"moe_balance_loss: {moe_balance_loss(g).item():.4f}")
    # print(f"Prompt weight shape: {aux['prompt_weight'].shape}")
    print(f"Number of parameters: {model.num_parameters}")

    # params and flops estimation
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Parameters: {params / 1e6:.2f} M")
    
