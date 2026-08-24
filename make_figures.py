"""make_figures.py — report figures from results/*.csv. Run from repo root."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "axes.axisbelow": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

CLEAN, DEVICE, TIME = ("#3a3a3a", "#8c8c8c", "#c0c0c0")


def tag(fs):
    return f"binary_rf_{fs}_baseline_n40"


# --- Fig 1: per-subject balanced accuracy -----------------------------------
c = pd.read_csv(R / f"{tag('clean')}_folds.csv")
d = pd.read_csv(R / f"{tag('device')}_folds.csv")
m = c.merge(d, on="subject", suffixes=("_clean", "_device")).sort_values("balanced_acc_clean")
x = np.arange(len(m))
fig, ax = plt.subplots(figsize=(6.5, 3.0))
ax.bar(x - 0.2, m.balanced_acc_clean, 0.4, label="Reference (29 features)", color=CLEAN)
ax.bar(x + 0.2, m.balanced_acc_device, 0.4, label="Device (15 features)", color=DEVICE)
ax.axhline(0.5, color="k", ls="--", lw=0.8)
ax.text(len(m) - 0.4, 0.515, "chance", ha="right", fontsize=7.5)
ax.axhline(m.balanced_acc_clean.mean(), color=CLEAN, ls=":", lw=1.0)
ax.set_xticks(x); ax.set_xticklabels(m.subject)
ax.set_ylabel("Balanced accuracy"); ax.set_xlabel("Held-out subject")
ax.set_ylim(0.4, 1.02); ax.legend(frameon=False, loc="upper left", fontsize=8)
fig.savefig(OUT / "fig_per_subject_balanced_accuracy.png"); plt.close(fig)

# --- Fig 2: confusion matrices ----------------------------------------------
sets = [("clean", "Reference, 29 features"), ("device", "Device, 15 features"),
        ("time_only", "Elapsed time only, 1 feature")]
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
for ax, (fs, title) in zip(axes, sets):
    cm = pd.read_csv(R / f"{tag(fs)}_confusion.csv", index_col=0)
    pct = cm.to_numpy() / cm.to_numpy().sum(axis=1, keepdims=True)
    ax.imshow(pct, cmap="Greys", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{pct[i,j]*100:.1f}%\n{cm.iloc[i,j]}", ha="center", va="center",
                    fontsize=7.5, color="white" if pct[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["non-stress", "stress"], fontsize=7.5)
    ax.set_yticklabels(["non-stress", "stress"], fontsize=7.5, rotation=90, va="center")
    ax.set_title(title, fontsize=8); ax.grid(False)
    ax.set_xlabel("Predicted", fontsize=8)
axes[0].set_ylabel("Actual", fontsize=8)
fig.savefig(OUT / "fig_confusion_matrices.png"); plt.close(fig)

# --- Fig 3: feature importance ----------------------------------------------
imp = pd.read_csv(R / f"{tag('clean')}_importance.csv").head(15).iloc[::-1]


def block(f):
    if f.startswith("eda"):
        return "EDA", CLEAN
    if f.startswith("hr"):
        return "HR", DEVICE
    if f.startswith("acc") or f.startswith("motion"):
        return "IMU", TIME
    return "Cross", "#e0e0e0"


colors = [block(f)[1] for f in imp.feature]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.barh(imp.feature, imp.importance_mean, xerr=imp.importance_sd,
        color=colors, edgecolor="k", linewidth=0.4, error_kw={"lw": 0.6})
ax.set_xlabel("Mean importance across 15 folds")
ax.tick_params(axis="y", labelsize=7.5)
handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="k", lw=0.4) for c in (CLEAN, DEVICE, TIME)]
ax.legend(handles, ["EDA", "HR", "IMU"], frameon=False, fontsize=8, loc="lower right")
fig.savefig(OUT / "fig_feature_importance.png"); plt.close(fig)

# --- Fig 4: configuration comparison ----------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 3.0))
labels, means, colors2 = [], [], []
for i, (fs, lab, col) in enumerate([("clean", "Reference\n29 features", CLEAN),
                                    ("device", "Device\n15 features", DEVICE),
                                    ("time_only", "Time only\n1 feature", TIME)]):
    f = pd.read_csv(R / f"{tag(fs)}_folds.csv")
    ax.bar(i, f.balanced_acc.mean(), 0.55, yerr=f.balanced_acc.std(),
           color=col, edgecolor="k", linewidth=0.5, capsize=3, error_kw={"lw": 0.8})
    ax.scatter(np.full(len(f), i) + np.random.uniform(-0.12, 0.12, len(f)),
               f.balanced_acc, s=9, color="k", zorder=3, alpha=0.6)
    ax.text(i, f.balanced_acc.mean() + f.balanced_acc.std() + 0.02,
            f"{f.balanced_acc.mean():.3f}", ha="center", fontsize=8)
    labels.append(lab)
ax.axhline(0.5, color="k", ls="--", lw=0.8)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Balanced accuracy"); ax.set_ylim(0.4, 1.08)
fig.savefig(OUT / "fig_configuration_comparison.png"); plt.close(fig)

print("wrote:", *[p.name for p in sorted(OUT.iterdir())], sep="\n  ")