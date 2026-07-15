"""
Fase 2: Normality test of Kolmogorov-Smirnov.
Tests blood_mg_dl and all 45 input features.

Outputs:
  __plots__/explore/csv/04_target_normality.csv
  __plots__/explore/csv/ks_test_results.csv
  __plots__/explore/png/04_target_normality.png
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_academic_style

warnings.filterwarnings("ignore")


def test_target_normality(series: pd.Series) -> dict:
    """Lilliefors-corrected KS test on blood_mg_dl, returns test result plus recommended_corr."""
    x = series.dropna().to_numpy()
    n = len(x)
    if n < 3:
        raise ValueError(f"Need at least 3 non-null observations, got {n}")

    stat_ks, p_ks = lilliefors(x, dist="norm")
    ks_normal = p_ks >= 0.05

    return {
        "n": n,
        "ks_lilliefors_stat": round(float(stat_ks), 6),
        "ks_lilliefors_p": round(float(p_ks), 6),
        "ks_lilliefors_normal": "Yes" if ks_normal else "No",
        "recommended_corr": "Pearson" if ks_normal else "Spearman",
    }


def test_feature_normality_ks(series: pd.Series) -> tuple[float, float]:
    """Lilliefors-corrected KS test for one feature. Returns (nan, nan) if constant."""
    x = series.dropna().to_numpy()
    if np.std(x, ddof=1) == 0:
        return float("nan"), float("nan")
    stat, p = lilliefors(x, dist="norm")
    return float(stat), float(p)

DATA_PATH = "__data__/neo/out/training_engineered.csv"
OUT_DIR = "__plots__/explore"
CSV_DIR = os.path.join(OUT_DIR, "csv")
PNG_DIR = os.path.join(OUT_DIR, "png")

RESPONSE = "blood_mg_dl"
EXCLUDE = ["patient_id", "is_augmented", "jaundice_label", RESPONSE]
ENGINEERED_PREFIXES = ("mean_zones_", "grad_z3z1_", "log1p_")
ENGINEERED_SUFFIXES = ("_R_div_B", "_G_minus_B", "_ITA")

ACCENT = "#0984e3"
RED = "#d63031"

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)
apply_academic_style()

def is_engineered(col):
    return any(col.startswith(p) for p in ENGINEERED_PREFIXES) or any(col.endswith(s) for s in ENGINEERED_SUFFIXES)

df = pd.read_csv(DATA_PATH)
FEATURES = [c for c in df.columns if c not in EXCLUDE and not is_engineered(c)]
assert len(FEATURES) == 45, f"Feature count is {len(FEATURES)}, not 45. Check columns: {FEATURES}"

df_orig = df[df["is_augmented"] == False].copy()
df_clean = df_orig.dropna(subset=FEATURES + [RESPONSE]).copy()

tsb = df_clean[RESPONSE].dropna()
norm_result = test_target_normality(tsb)
stat_ks, p_ks = norm_result["ks_lilliefors_stat"], norm_result["ks_lilliefors_p"]

pd.DataFrame([norm_result]).T.reset_index().rename(
    columns={"index": "statistic", 0: "value"}
).to_csv(os.path.join(CSV_DIR, "04_target_normality.csv"), index=False)

fig, ax = plt.subplots(figsize=(6, 5))
(osm, osr), (slope, intercept, _) = stats.probplot(tsb, dist="norm")
ax.scatter(osm, osr, s=14, color=ACCENT, alpha=0.6, edgecolors="none", label="Sample quantiles")
ref_line = slope * np.array([osm[0], osm[-1]]) + intercept
ax.plot([osm[0], osm[-1]], ref_line, color=RED, linewidth=1.8, linestyle="--", label="Normal reference line")
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Sample Quantiles")
ax.set_title(
    f"Q-Q Plot: blood_mg_dl\n"
    f"Lilliefors-KS: D={stat_ks:.4f}, p={p_ks:.4f} ({'Non-normal' if p_ks < 0.05 else 'Normal'})"
)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PNG_DIR, "04_target_normality.png"))
plt.close()

results = []
for col in FEATURES:
    series = df_clean[col]
    x = series.dropna().values
    mu, sigma = np.mean(x), np.std(x, ddof=1)
    stat, p = test_feature_normality_ks(series)
    results.append({
        "feature": col,
        "n": len(x),
        "mean": mu,
        "std": sigma,
        "skewness": stats.skew(x),
        "excess_kurtosis": stats.kurtosis(x),
        "KS_statistic": stat,
        "KS_pvalue": p,
        "normal_alpha_0.05": "Normal" if (not np.isnan(p) and p > 0.05) else "Non-Normal",
    })

res_df = pd.DataFrame(results).sort_values("KS_pvalue", ascending=False).reset_index(drop=True)

tsb_row = {
    "feature": RESPONSE,
    "n": norm_result["n"],
    "mean": round(float(tsb.mean()), 4),
    "std": round(float(tsb.std(ddof=1)), 4),
    "skewness": round(float(stats.skew(tsb)), 4),
    "excess_kurtosis": round(float(stats.kurtosis(tsb)), 4),
    "KS_statistic": round(stat_ks, 6),
    "KS_pvalue": round(p_ks, 6),
    "normal_alpha_0.05": "Normal" if norm_result["ks_lilliefors_normal"] == "Yes" else "Non-Normal",
}

combined_df = pd.concat([res_df, pd.DataFrame([tsb_row])], ignore_index=True)
combined_df.to_csv(os.path.join(CSV_DIR, "ks_test_results.csv"), index=False)

n_normal = (res_df["normal_alpha_0.05"] == "Normal").sum()
print(f"phase2 done: {n_normal}/{len(res_df)} features normal")