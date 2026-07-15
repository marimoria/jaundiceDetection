"""
Correlation Analysis:
  1. Every feature vs blood_mg_dl (Spearman)
  2. Feature-vs-feature redundancy within each zone (14x14 matrix)
  3. Cross-zone comparison of top-5 features (Kramer's rule test)

All analysis runs exclusively on original (non-augmented) rows.

Outputs:
  __plots__/explore/
    csv/
      04_layer_a_feature_target_corr.csv
      04_layer_a_kde_separation.csv
      05_layer_b_intrazone_redundancy.csv
      06_layer_c_crosszone_comparison.csv
    png/
      04_layer_a_spearman_bar.png
      04_layer_a_kde_top5.png
      05_layer_b_heatmap_zone1.png
      05_layer_b_heatmap_zone2.png
      05_layer_b_heatmap_zone3.png
      06_layer_c_crosszone.png

Usage:
  python phase3_correlation.py
"""

import logging
import os
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_academic_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("phase3")

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_PATH = "__data__/neo/out/training_engineered.csv"
OUT_DIR   = "__plots__/explore"
CSV_DIR   = os.path.join(OUT_DIR, "csv")
PNG_DIR   = os.path.join(OUT_DIR, "png")

TARGET = "blood_mg_dl"

ACCENT  = "#0984e3"
RED     = "#d63031"
GREEN   = "#00b894"
GRAY    = "#636e72"
ORANGE  = "#e17055"

REDUNDANCY_THRESHOLD = 0.90

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

apply_academic_style()

df_all = pd.read_csv(DATA_PATH)
orig   = df_all[df_all["is_augmented"] == False].copy()

ZONE_FEATURES = [c for c in df_all.columns if c.startswith("zone")]
META_FEATURES = ["gestational_age", "postnatal_age_days", "weight"]
ALL_FEATURES  = ZONE_FEATURES + META_FEATURES

ZONES = ["zone1", "zone2", "zone3"]
ZONE_FEATURE_NAMES = sorted({
    "_".join(c.split("_")[1:]) for c in ZONE_FEATURES
})

log.info("Loaded %d original rows, %d features", len(orig), len(ALL_FEATURES))


def spearman(series_a, series_b):
    mask = series_a.notna() & series_b.notna()
    a, b = series_a[mask], series_b[mask]
    rsp, psp = stats.spearmanr(a, b)
    return float(rsp), float(psp) # type: ignore


# Feature vs target

log.info("Layer A: Feature vs target")

rows_a = []
for feat in ALL_FEATURES:
    if feat not in orig.columns:
        continue
    rsp, psp = spearman(orig[feat], orig[TARGET])
    rows_a.append({
        "feature":        feat,
        "spearman_r":     round(rsp, 4),
        "spearman_p":     round(psp, 6),
        "abs_spearman":   round(abs(rsp), 4),
        "near_zero":      abs(rsp) < 0.10,
    })

layer_a = pd.DataFrame(rows_a).sort_values("abs_spearman", ascending=False)
layer_a.to_csv(os.path.join(CSV_DIR, "04_layer_a_feature_target_corr.csv"), index=False)

top5    = layer_a.head(5)["feature"].tolist()
bottom5 = layer_a[layer_a["near_zero"]]["feature"].tolist()
log.info("  Top-5 by |Spearman|: %s", top5)
log.info("  Near-zero features (|r|<0.10): %s", bottom5)

fig, ax = plt.subplots(figsize=(14, 10))
colors = []
for _, row in layer_a.iterrows():
    if abs(row["spearman_r"]) < 0.10:
        colors.append(GRAY)
    elif abs(row["spearman_r"]) >= 0.50:
        colors.append(GREEN)
    elif abs(row["spearman_r"]) >= 0.30:
        colors.append(ACCENT)
    else:
        colors.append(ORANGE)

ax.barh(layer_a["feature"].tolist(), layer_a["spearman_r"].tolist(),
        color=colors, edgecolor="white", height=0.72, alpha=0.88)
ax.axvline(0,     color="#2d3436", linewidth=1.0)
ax.axvline( 0.50, color=GREEN, linewidth=1.2, linestyle="--", alpha=0.7)
ax.axvline(-0.50, color=GREEN, linewidth=1.2, linestyle="--", alpha=0.7)
ax.axvline( 0.30, color=ACCENT, linewidth=1.0, linestyle=":", alpha=0.7)
ax.axvline(-0.30, color=ACCENT, linewidth=1.0, linestyle=":", alpha=0.7)

from matplotlib.patches import Patch
legend_patches = [
    Patch(color=GREEN,  label="|r| ≥ 0.50 (strong)"),
    Patch(color=ACCENT, label="0.30 ≤ |r| < 0.50 (moderate)"),
    Patch(color=ORANGE, label="0.10 ≤ |r| < 0.30 (weak)"),
    Patch(color=GRAY,   label="|r| < 0.10 (near-zero)"),
]
ax.legend(handles=legend_patches, loc="lower right")
ax.set_xlabel("Spearman r with blood_mg_dl")
ax.set_title("Layer A: Spearman Correlation — All Features vs blood_mg_dl (original rows only)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(PNG_DIR, "04_layer_a_spearman_bar.png"))
plt.close()
log.info("  Saved 04_layer_a_spearman_bar.png")


# Layer A (cont.): KDE distribution by class for top-5 features

log.info("Layer A cont.: KDE separation for top-5 features")

from scipy.stats import gaussian_kde, ks_2samp

kde_rows = []
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes_flat = axes.flatten()

for i, feat in enumerate(top5):
    ax = axes_flat[i]
    sub = orig[[feat, "jaundice_label"]].dropna()
    g0 = sub.loc[sub["jaundice_label"] == 0, feat].to_numpy()
    g1 = sub.loc[sub["jaundice_label"] == 1, feat].to_numpy()

    x_min = min(g0.min(), g1.min())
    x_max = max(g0.max(), g1.max())
    x_grid = np.linspace(x_min, x_max, 400)

    kde0 = gaussian_kde(g0)
    kde1 = gaussian_kde(g1)
    y0 = kde0(x_grid)
    y1 = kde1(x_grid)

    overlap = float(np.trapezoid(np.minimum(y0, y1), x_grid))
    ks_stat, ks_p = ks_2samp(g0, g1)

    kde_rows.append({
        "feature":             feat,
        "normal_mean":         round(float(g0.mean()), 4),
        "jaundice_mean":       round(float(g1.mean()), 4),
        "overlap_coefficient": round(overlap, 4),
        "ks_2samp_stat":       round(float(ks_stat), 4),
        "ks_2samp_p":          round(float(ks_p), 6),
    })

    ax.fill_between(x_grid, y0, color=ACCENT, alpha=0.45)
    ax.fill_between(x_grid, y1, color=RED, alpha=0.45)
    ax.plot(x_grid, y0, color=ACCENT, linewidth=1.8)
    ax.plot(x_grid, y1, color=RED, linewidth=1.8)
    ax.text(
        0.97, 0.95, f"overlap={overlap:.2f}\nKS D={ks_stat:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
    )
    ax.set_title(feat, fontsize=10)
    if i % 3 == 0:
        ax.set_ylabel("Density")

for j in range(len(top5), len(axes_flat)):
    fig.delaxes(axes_flat[j])

legend_patches_kde = [
    Patch(color=ACCENT, label="Normal"),
    Patch(color=RED,    label="Jaundiced"),
]
fig.legend(handles=legend_patches_kde, loc="upper right", ncol=1)
fig.suptitle(
    "Layer A (cont.): KDE Distribution by Class — Top-5 Features\n"
    "(lower overlap coefficient = clearer class separation)",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(os.path.join(PNG_DIR, "04_layer_a_kde_top5.png"))
plt.close()

kde_df = pd.DataFrame(kde_rows).sort_values("overlap_coefficient")
kde_df.to_csv(os.path.join(CSV_DIR, "04_layer_a_kde_separation.csv"), index=False)
log.info("  Saved 04_layer_a_kde_top5.png")
log.info(
    "  Overlap coefficients: %s",
    kde_df.set_index("feature")["overlap_coefficient"].round(3).to_dict(),
)


# Intra-zone redundancy heatmaps

log.info("Layer B: intra-zone feature redundancy")

redundancy_rows = []

for zone in ZONES:
    zone_cols = [c for c in ZONE_FEATURES if c.startswith(zone + "_")]
    short_names = ["_".join(c.split("_")[1:]) for c in zone_cols]

    n = len(zone_cols)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mask = orig[zone_cols[i]].notna() & orig[zone_cols[j]].notna()
            r, _ = stats.spearmanr(orig.loc[mask, zone_cols[i]],
                                   orig.loc[mask, zone_cols[j]])
            mat[i, j] = float(r) # type: ignore
            mat[j, i] = float(r) # type: ignore
            if abs(r) >= REDUNDANCY_THRESHOLD: # type: ignore
                redundancy_rows.append({
                    "zone":    zone,
                    "feat_a":  zone_cols[i],
                    "feat_b":  zone_cols[j],
                    "spearman_r": round(float(r), 4), # type: ignore
                })

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman r")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)

    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    ax.set_title(f"Layer B: Intra-zone Spearman Correlation — {zone} (|r|≥{REDUNDANCY_THRESHOLD} flagged)")
    plt.tight_layout()
    out_path = os.path.join(PNG_DIR, f"05_layer_b_heatmap_{zone}.png")
    plt.savefig(out_path)
    plt.close()
    log.info("  Saved 05_layer_b_heatmap_%s.png", zone)

redundancy_df = pd.DataFrame(redundancy_rows).sort_values("spearman_r", key=abs, ascending=False)
redundancy_df.to_csv(os.path.join(CSV_DIR, "05_layer_b_intrazone_redundancy.csv"), index=False)
log.info("  Redundant pairs (|r|≥%.2f): %d", REDUNDANCY_THRESHOLD, len(redundancy_df))


# Layer C: cross-zone comparison (Kramer's rule)

log.info("Layer C: cross-zone comparison of top-5 features")

KEY_FEATURES = []
for feat in top5:
    if feat in ZONE_FEATURES:
        suffix = feat.split("_", 1)[1]
        if suffix not in KEY_FEATURES:
            KEY_FEATURES.append(suffix)

crosszone_rows = []
for feat_suffix in KEY_FEATURES:
    for zone in ZONES:
        col = f"{zone}_{feat_suffix}"
        if col not in orig.columns:
            continue
        rsp, psp = spearman(orig[col], orig[TARGET])
        crosszone_rows.append({
            "feature_type": feat_suffix,
            "zone":         zone,
            "column":       col,
            "spearman_r":   round(rsp, 4),
            "spearman_p":   round(psp, 6),
            "abs_spearman": round(abs(rsp), 4),
        })

crosszone_df = pd.DataFrame(crosszone_rows)
crosszone_df.to_csv(os.path.join(CSV_DIR, "06_layer_c_crosszone_comparison.csv"), index=False)

for feat_suffix in KEY_FEATURES:
    sub = crosszone_df[crosszone_df["feature_type"] == feat_suffix]
    vals = sub.set_index("zone")["spearman_r"]
    log.info("  %s — zone1: %.3f  zone2: %.3f  zone3: %.3f",
             feat_suffix,
             vals.get("zone1", float("nan")),
             vals.get("zone2", float("nan")),
             vals.get("zone3", float("nan")))

fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
axes_flat = axes.flatten()
zone_colors = {"zone1": ACCENT, "zone2": GREEN, "zone3": RED}

for i, feat_suffix in enumerate(KEY_FEATURES):
    ax = axes_flat[i]
    sub = crosszone_df[crosszone_df["feature_type"] == feat_suffix].copy()
    sub = sub.set_index("zone").reindex(ZONES).reset_index()
    bars = ax.bar(
        sub["zone"], sub["spearman_r"],
        color=[zone_colors[z] for z in sub["zone"]],
        edgecolor="white", alpha=0.88, width=0.5,
    )
    for bar, val in zip(bars, sub["spearman_r"]):
        y_pos = val + 0.015 if val >= 0 else val - 0.03
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="#2d3436", linewidth=0.8)
    ax.set_title(feat_suffix, fontsize=10)
    if i % 3 == 0:
        ax.set_ylabel("Spearman r")
    ax.set_ylim(
        min(sub["spearman_r"].min() - 0.1, -0.05),
        max(sub["spearman_r"].max() + 0.12,  0.05),
    )
    ax.xaxis.set_tick_params(labelsize=9)

for j in range(len(KEY_FEATURES), len(axes_flat)):
    fig.delaxes(axes_flat[j])

from matplotlib.patches import Patch as _Patch
legend_patches_c = [_Patch(color=zone_colors[z], label=z) for z in ZONES]
fig.legend(handles=legend_patches_c, loc="upper right", ncol=1)
fig.suptitle("Layer C: Cross-zone Spearman r for Top-5 Features vs blood_mg_dl\n"
             "(zone3 > zone1 confirms Kramer's cephalocaudal rule)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PNG_DIR, "06_layer_c_crosszone.png"))
plt.close()
log.info("  Saved 06_layer_c_crosszone.png")


log.info("Phase 3 complete. Outputs in %s/", OUT_DIR)