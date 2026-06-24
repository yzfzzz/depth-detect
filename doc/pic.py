import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ============================================================
# Data — combined
# ============================================================

latency_groups = [
    ("E2E\nONNX/TRT",     133.0, 5.77,  "ONNX (CPU)",      "TensorRT (GPU)"),
    ("E2E\nSync/Async",   5.77,  5.35,  "TRT Sync",        "TRT Overlap"),
    ("YOLO\nPreprocess",  6.81,  0.601, "OpenCV (CPU)",    "CUDA (GPU)"),
    ("YOLO\nPostprocess", 2.35,  0.461, "OpenCV (CPU)",    "CUDA (GPU)"),
    ("Depth\nPreprocess", 1.42,  0.376, "OpenCV (CPU)",    "CUDA (GPU)"),
    ("Depth\nPostprocess",1.39,  0.783, "OpenCV (CPU)",    "CUDA (GPU)"),
]

speedup_data = [
    ("E2E\nONNX/TRT",     23.1),
    ("E2E\nSync/Async",    1.08),
    ("YOLO\nPreprocess",  11.3),
    ("YOLO\nPostprocess",  5.1),
    ("Depth\nPreprocess",  3.8),
    ("Depth\nPostprocess", 1.8),
]

# ============================================================
# Layout: 2 rows × 1 col
# ============================================================
fig, (ax_lat, ax_spd) = plt.subplots(2, 1, figsize=(22, 14),
                                      gridspec_kw={'height_ratios': [1.2, 1]})

BLUE   = '#4A90D9'
ORANGE = '#E87D2F'
GRAY   = '#888888'

x = np.arange(len(latency_groups))
w = 0.30
gap_idx = 1.5

# ============================================================
# Upper: Latency (log scale)
# ============================================================
colors_a = [BLUE,   '#D4772A', BLUE,   BLUE,   BLUE,   BLUE]
colors_b = [ORANGE, '#F0A355', ORANGE, ORANGE, ORANGE, ORANGE]
vals_a   = [g[1] for g in latency_groups]
vals_b   = [g[2] for g in latency_groups]

bars_a = ax_lat.bar(x - w/2, vals_a, w, color=colors_a, edgecolor='white', linewidth=0.5)
bars_b = ax_lat.bar(x + w/2, vals_b, w, color=colors_b, edgecolor='white', linewidth=0.5)

ax_lat.set_yscale('log')
ax_lat.set_ylabel('Latency (ms, log scale)', fontsize=18)
ax_lat.set_title('GeForce5060 Latency Comparison', fontsize=20, fontweight='bold', pad=15)
ax_lat.set_xticks(x)
ax_lat.set_xticklabels([g[0] for g in latency_groups], fontsize=14)
ax_lat.tick_params(axis='y', labelsize=13)
ax_lat.grid(axis='y', alpha=0.3, linestyle='--')
ax_lat.set_ylim(0.1, 400)

# Category separator
ax_lat.axvline(x=gap_idx, color=GRAY, linestyle=':', linewidth=1.5, alpha=0.6)
ax_lat.text(0.5, 0.97, 'End-to-End Pipeline', transform=ax_lat.transAxes,
            ha='center', fontsize=14, fontweight='bold', color='#555555',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', alpha=0.85))
ax_lat.text(0.82, 0.97, 'Pre/Post Processing', transform=ax_lat.transAxes,
            ha='center', fontsize=14, fontweight='bold', color='#555555',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', alpha=0.85))

# Value labels
for bar, val in zip(bars_a, vals_a):
    y_offset = val * 0.35 if val < 10 else val * 0.18
    ax_lat.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset,
                f'{val:.2f}' if val < 10 else f'{val:.0f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=bar.get_facecolor())

for bar, val in zip(bars_b, vals_b):
    y_offset = val * 0.35 if val < 10 else val * 0.18
    ax_lat.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset,
                f'{val:.2f}' if val < 10 else f'{val:.0f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=bar.get_facecolor())

# Legend — "End-to-End Pipeline" 文字下方
legend_elements = [
    Patch(facecolor=BLUE,      label='ONNX / OpenCV (CPU)'),
    Patch(facecolor=ORANGE,    label='TensorRT / CUDA (GPU)'),
    Patch(facecolor='#D4772A', label='TRT Sync'),
    Patch(facecolor='#F0A355', label='TRT Overlap'),
]
# ax_lat.legend(handles=legend_elements, fontsize=13, framealpha=0.95,
#               ncol=4, loc='upper center', bbox_to_anchor=(0.22, 0.92))

ax_lat.legend(handles=legend_elements, fontsize=13, framealpha=0.95,
              ncol=4, loc='upper center', bbox_to_anchor=(0.5, 0.90))

# ============================================================
# Lower: Speedup (log scale)
# ============================================================
labels_spd = [s[0] for s in speedup_data]
vals_spd   = [s[1] for s in speedup_data]
colors_spd = ['#C0392B' if v > 10 else ('#E67E22' if v > 5 else '#3FA363') for v in vals_spd]

bars_spd = ax_spd.bar(x, vals_spd, w * 1.4, color=colors_spd, edgecolor='white', linewidth=0.5)
ax_spd.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, label='1× baseline')

ax_spd.set_ylabel('Speedup', fontsize=18)
ax_spd.set_title('GeForce5060 Speedup Comparison', fontsize=20, fontweight='bold', pad=15)
ax_spd.set_xticks(x)
ax_spd.set_xticklabels(labels_spd, fontsize=14)
ax_spd.tick_params(axis='y', labelsize=13)
ax_spd.grid(axis='y', alpha=0.3, linestyle='--')
ax_spd.set_yscale('log')

# Category separator
ax_spd.axvline(x=gap_idx, color=GRAY, linestyle=':', linewidth=1.5, alpha=0.6)
ax_spd.text(0.5, 0.97, 'End-to-End Pipeline', transform=ax_spd.transAxes,
            ha='center', fontsize=14, fontweight='bold', color='#555555',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', alpha=0.85))
ax_spd.text(0.82, 0.97, 'Pre/Post Processing', transform=ax_spd.transAxes,
            ha='center', fontsize=14, fontweight='bold', color='#555555',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', alpha=0.85))

# Legend — 右上角
ax_spd.legend(fontsize=13, framealpha=0.95, loc='upper right')

# Speedup value labels
for bar, val in zip(bars_spd, vals_spd):
    y_off = val * 0.22
    ax_spd.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_off,
                f'{val:.1f}×' if val >= 10 else f'{val:.2f}×',
                ha='center', fontsize=12, fontweight='bold', color=bar.get_facecolor())

# ============================================================
plt.tight_layout(pad=3)
plt.savefig('benchmark_combined.png', dpi=150, bbox_inches='tight')
plt.show()