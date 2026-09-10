import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

FP32_COLOR = "#2E6F95"
INT8_COLOR = "#C97A24"
INK = "#20242A"
GRID = "#D7DAD5"
BG = "#F7F7F5"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
})


def grouped_bar(labels, series, ylabel, title, out_name, ymax=None, figsize=(8, 4.5), fmt="{:.3f}", legend_loc="upper right"):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    n = len(series)
    x = np.arange(len(labels))
    width = 0.8 / n
    # extra headroom above the tallest bar+label so the legend never sits on top of a value label
    peak = max(max(s[1]) for s in series)
    effective_ymax = ymax or peak * 1.25
    for i, (name, values, color, alpha) in enumerate(series):
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, values, width * 0.92, label=name, color=color, alpha=alpha, zorder=3)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + effective_ymax * 0.02,
                    fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=14)
    ax.set_ylim(0, effective_ymax)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc=legend_loc)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, out_name))
    plt.close(fig)
    print("wrote", out_name)


# --- Task 1: per-class AP ---
classes = ["person", "bicycle", "car", "traffic\nlight", "stop\nsign"]
ap50 = [0.7751, 0.4081, 0.5937, 0.4361, 0.7178]
ap5095 = [0.5291, 0.2446, 0.3797, 0.2326, 0.6872]
grouped_bar(
    classes,
    [("AP@0.5", ap50, FP32_COLOR, 1.0), ("AP@0.5:0.95", ap5095, FP32_COLOR, 0.45)],
    "Average Precision", "Task 1 — FP32 baseline, per-class AP", "task1_per_class_ap.png", ymax=0.95,
)

# --- Task 2: AP by object size, FP32 vs INT8 ---
sizes = ["Small\n(<32²px)", "Medium\n(32²–96²px)", "Large\n(>96²px)"]
fp32_size = [0.2045, 0.5978, 0.7186]
int8_size = [0.1658, 0.5730, 0.7518]
grouped_bar(
    sizes,
    [("FP32", fp32_size, FP32_COLOR, 1.0), ("INT8", int8_size, INT8_COLOR, 1.0)],
    "AP@[.5:.95]", "Task 2 — AP by object size, FP32 vs INT8", "task2_ap_by_size.png", ymax=0.9,
    legend_loc="upper left",
)

# --- Task 2: per-class AP delta ---
grouped_bar(
    classes,
    [("FP32", ap5095, FP32_COLOR, 1.0), ("INT8", [0.5111, 0.2356, 0.3704, 0.2138, 0.6411], INT8_COLOR, 1.0)],
    "AP@[.5:.95]", "Task 2 — Per-class AP, FP32 vs INT8", "task2_per_class_ap.png", ymax=0.8,
)

# --- Task 2: degradation 5x2 table, mAP@0.5 ---
conditions = ["Low\nlight", "Clean", "JPEG\nq30", "Downscale/\nupscale", "Motion\nblur"]
fp32_deg = [0.5155, 0.5862, 0.5337, 0.5426, 0.2807]
int8_deg = [0.5124, 0.5688, 0.5145, 0.5167, 0.2502]
grouped_bar(
    conditions,
    [("FP32", fp32_deg, FP32_COLOR, 1.0), ("INT8", int8_deg, INT8_COLOR, 1.0)],
    "mAP@0.5", "Task 2 — mAP@0.5 across degradations, FP32 vs INT8", "task2_degradation_map50.png",
    ymax=0.68, figsize=(9, 4.8),
)

# --- Task 2: relative INT8 gap per condition (diverging horizontal bars) ---
fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
gap_labels = ["Low light", "Clean", "JPEG q30", "Downscale/upscale", "Motion blur"]
gap_vals = [-0.6, -3.0, -3.6, -4.8, -10.9]
colors = ["#3F8F5F" if v > -3.5 else "#BD4A3B" for v in gap_vals]
y = np.arange(len(gap_labels))
bars = ax.barh(y, gap_vals, color=colors, height=0.55, zorder=3)
# all value labels sit in one column to the right of x=0 (empty region, since every
# bar is <=0) so a long bar (motion blur) never pushes its label into the y-axis
# category labels on the left.
xmin, label_x = min(gap_vals) * 1.18, 0.35
for b, v in zip(bars, gap_vals):
    ax.text(label_x, b.get_y() + b.get_height() / 2, f"{v:+.1f}%",
            va="center", ha="left", fontsize=10, color=INK)
ax.set_yticks(y)
ax.set_yticklabels(gap_labels, fontsize=10)
ax.set_xlim(xmin, 4.5)
ax.axvline(0, color=INK, linewidth=1)
ax.set_xlabel("INT8 gap relative to FP32's own score in that condition (%)", fontsize=10)
ax.set_title("Task 2 — INT8's relative accuracy gap, by condition", fontsize=12, fontweight="bold", loc="left", pad=14)
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "task2_relative_gap.png"))
plt.close(fig)
print("wrote task2_relative_gap.png")

# --- Task 3: calibrated scale ratios ---
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
layer_labels = [
    "model.0/conv", "model.0/act", "model.1/conv", "model.1/act",
    "model.2/cv1/conv", "model.2/split", "model.2/m0.cv1/conv", "model.2/m0.cv1/act",
    "model.2/m0.cv2/conv", "model.2/concat", "model.2/cv2/conv", "model.2/cv2/act",
][::-1]
ratios = [0.937, 0.917, 0.716, 0.662, 0.834, 0.844, 0.882, 0.777, 0.717, 0.620, 0.686, 0.529][::-1]
y = np.arange(len(layer_labels))
bars = ax.barh(y, ratios, color=INT8_COLOR, height=0.6, zorder=3)
for b, v in zip(bars, ratios):
    ax.text(v + 0.015, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9, color=INK)
ax.axvline(1.0, color=INK, linewidth=1.2, linestyle="--")
ax.text(1.0, len(layer_labels) - 0.2, " ratio = 1.0 (unchanged)", fontsize=9, color=INK, va="bottom")
ax.set_yticks(y)
ax.set_yticklabels(layer_labels, fontsize=9)
ax.set_xlabel("Calibrated scale ratio: blur-calibrated ÷ clean-calibrated", fontsize=10)
ax.set_title("Task 3 — Blur-calibration made scales tighter (95% of 124 tensors)", fontsize=12, fontweight="bold", loc="left", pad=14)
ax.set_xlim(0, 1.15)
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "task3_scale_ratios.png"))
plt.close(fig)
print("wrote task3_scale_ratios.png")
