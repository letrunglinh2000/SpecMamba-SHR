"""
Visualize the physics-grounded diffusion training process.
Usage: python visualize_diffusion.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

T = 8
mask_alpha_thresh = 0.3

# Cosine schedule: alpha_t = 0.5 * (1 - cos(pi * t / T))
t_vals = np.arange(0, T + 1)
alphas = 0.5 * (1 - np.cos(np.pi * t_vals / T))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Physics-Grounded Diffusion Training (T=8)", fontsize=14, fontweight='bold')

# ── Panel 1: Alpha schedule ──────────────────────────────────────────────────
ax = axes[0]
ax.plot(t_vals, alphas, 'o-', color='steelblue', linewidth=2, markersize=8, zorder=3)
ax.axhline(y=mask_alpha_thresh, color='tomato', linestyle='--', linewidth=1.5,
           label=f'mask_alpha_thresh = {mask_alpha_thresh}')
ax.fill_between(t_vals, alphas, alpha=0.15, color='steelblue')

for t, a in zip(t_vals, alphas):
    ax.annotate(f'{a:.3f}', (t, a), textcoords='offset points',
                xytext=(0, 10), ha='center', fontsize=8)

ax.set_xlabel('Timestep t', fontsize=11)
ax.set_ylabel('α_t  (specular fraction)', fontsize=11)
ax.set_title('Cosine Alpha Schedule', fontsize=12)
ax.set_xticks(t_vals)
ax.set_xticklabels([f't={i}' for i in t_vals], rotation=30, ha='right')
ax.set_ylim(-0.05, 1.15)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# annotate t=0 and t=T
ax.annotate('x₀ ≈ H\n(clean)', (0, alphas[0]), textcoords='offset points',
            xytext=(15, -25), fontsize=9, color='green',
            arrowprops=dict(arrowstyle='->', color='green'))
ax.annotate('x_T = L\n(specular input)', (T, alphas[T]), textcoords='offset points',
            xytext=(-60, -30), fontsize=9, color='darkorange',
            arrowprops=dict(arrowstyle='->', color='darkorange'))

# ── Panel 2: What x_t looks like at each step ────────────────────────────────
ax = axes[1]

# Simulate a 1D signal: H=0.4 (dark), L=0.9 (bright with highlight)
H_val = 0.4
L_val = 0.9
x_t_vals = H_val + alphas * (L_val - H_val)

colors = plt.cm.RdYlGn_r(alphas)
bars = ax.bar(t_vals, x_t_vals, color=colors, edgecolor='gray', linewidth=0.5, zorder=3)

# Highlight region where mask loss is active
for i, (t, a, xt) in enumerate(zip(t_vals, alphas, x_t_vals)):
    label = 'mask OFF' if a <= mask_alpha_thresh else 'mask ON'
    color = 'lightcoral' if a <= mask_alpha_thresh else 'lightgreen'
    ax.bar(t, xt, color=color, edgecolor='gray', linewidth=0.5, zorder=3)
    ax.text(t, xt + 0.01, f'α={a:.2f}', ha='center', fontsize=7.5, rotation=45)

ax.axhline(y=H_val, color='green', linestyle=':', linewidth=1.5, label=f'H (clean) = {H_val}')
ax.axhline(y=L_val, color='darkorange', linestyle=':', linewidth=1.5, label=f'L (input) = {L_val}')

mask_off_patch = mpatches.Patch(color='lightcoral', label='mask loss OFF (α ≤ 0.3)')
mask_on_patch  = mpatches.Patch(color='lightgreen', label='mask loss ON  (α > 0.3)')
ax.legend(handles=[mask_off_patch, mask_on_patch,
                   mpatches.Patch(color='green', label='H clean target'),
                   mpatches.Patch(color='darkorange', label='L corrupted input')],
          fontsize=8, loc='lower right')

ax.set_xlabel('Timestep t', fontsize=11)
ax.set_ylabel('Pixel intensity of x_t', fontsize=11)
ax.set_title('x_t = H + α_t·(L−H)  at each t', fontsize=12)
ax.set_xticks(t_vals)
ax.set_xticklabels([f't={i}' for i in t_vals], rotation=30, ha='right')
ax.set_ylim(0.3, 1.05)
ax.grid(True, alpha=0.3, axis='y')

# ── Panel 3: Training procedure diagram ──────────────────────────────────────
ax = axes[2]
ax.axis('off')

procedure = [
    ("1. Sample t ~ Uniform[1, T]", "steelblue"),
    ("2. Compute α_t  (cosine schedule)", "steelblue"),
    ("", None),
    ("3. Build x_t = H + α_t · (L − H)", "darkorange"),
    ("   t=T  →  x_T = L  (fully corrupted)", "gray"),
    ("   t=1  →  x_1 ≈ H  (nearly clean)", "gray"),
    ("", None),
    ("4. SpecMamba(x_t)  →  spec_prob", "purple"),
    ("   (mask supervision when α_t > 0.3)", "gray"),
    ("", None),
    ("5. AnyIR(x_t, t_emb, spec_prob) → H_hat", "green"),
    ("", None),
    ("6. Losses:", "black"),
    ("   L_recon  = L1(H_hat, H)          ×1.0", "darkgreen"),
    ("   L_score  = α·||h_hat − h_gt||²   ×0.5", "teal"),
    ("   L_mask   = CE + Dice(mask, M_gt)  ×0.2", "mediumpurple"),
    ("", None),
    ("7. Inference: single pass at t=T", "tomato"),
    ("   H_final = AnyIR(L, t_T, spec_prob)", "tomato"),
]

y = 0.97
for text, color in procedure:
    if not text:
        y -= 0.03
        continue
    ax.text(0.02, y, text, transform=ax.transAxes,
            fontsize=9.5, color=color or 'black',
            fontfamily='monospace', verticalalignment='top')
    y -= 0.048

ax.set_title('Training Procedure', fontsize=12)
ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='lightgray',
                            linewidth=1, transform=ax.transAxes))

plt.tight_layout()
out_path = 'diffusion_visualization.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
plt.show()
