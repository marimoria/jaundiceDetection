"""
Normality Test (Kolmogorov-Smirnov)
=====================================================================================
Dataset        : __data__/neo/out/training_engineered.csv (Updated for GitHub portability)
Response Var   : TSB -> 'blood_mg_dl' column
Features Tested: 45 features (Skin color zones RGB/YCrCb/HSL/Lab + basic clinical data)
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# 1. LOAD DATA (Updated with Relative Path for GitHub)
# ---------------------------------------------------------------
# Using relative path so anyone cloning this repository can run it immediately
df = pd.read_csv("__data__/neo/out/training_engineered.csv")

RESPONSE = "blood_mg_dl"          # TSB (Total Serum Bilirubin)
EXCLUDE = ["patient_id", "is_augmented", "jaundice_label", RESPONSE]

# Only use original 45 features (no engineered features)
ENGINEERED_PREFIXES = ("mean_zones_", "grad_z3z1_", "log1p_")
ENGINEERED_SUFFIXES = ("_R_div_B", "_G_minus_B", "_ITA")

def is_engineered(col):
    return (
        any(col.startswith(p) for p in ENGINEERED_PREFIXES) or
        any(col.endswith(s) for s in ENGINEERED_SUFFIXES)
    )

FEATURES = [c for c in df.columns if c not in EXCLUDE and not is_engineered(c)]

print(f"Number of features being tested: {len(FEATURES)}")
assert len(FEATURES) == 45, f"Feature count is {len(FEATURES)}, not 45. Check columns: {FEATURES}"

# Only run on original (non-augmented) rows
df_orig = df[df["is_augmented"] == False].copy()

# Handle missing values -> drop rows with nulls in features or response
df_clean = df_orig.dropna(subset=FEATURES + [RESPONSE]).copy()
print(f"Rows used after dropping NA: {df_clean.shape[0]} out of {df_orig.shape[0]}")


# ---------------------------------------------------------------
# 2. KOLMOGOROV-SMIRNOV (KS) NORMALITY TEST FOR FEATURES
# ---------------------------------------------------------------
def ks_normality_test(series):
    x = series.dropna().values
    mu, sigma = np.mean(x), np.std(x, ddof=1)
    if sigma == 0:
        return np.nan, np.nan, mu, sigma
    stat, p = stats.kstest(x, 'norm', args=(mu, sigma))
    return stat, p, mu, sigma

results = []
for col in FEATURES:
    stat, p, mu, sigma = ks_normality_test(df_clean[col])
    skew = stats.skew(df_clean[col].dropna())
    kurt = stats.kurtosis(df_clean[col].dropna())  # excess kurtosis
    results.append({
        "feature": col,
        "n": df_clean[col].dropna().shape[0],
        "mean": mu,
        "std": sigma,
        "skewness": skew,
        "excess_kurtosis": kurt,
        "KS_statistic": stat,
        "KS_pvalue": p,
        "normal_alpha_0.05": "Normal" if p > 0.05 else "Non-Normal"
    })

res_df = pd.DataFrame(results).sort_values("KS_pvalue", ascending=False).reset_index(drop=True)

n_normal = (res_df["normal_alpha_0.05"] == "Normal").sum()
n_total = res_df.shape[0]
print(f"\n=== KOLMOGOROV-SMIRNOV TEST RESULTS (alpha=0.05) ===")
print(f"Features with NORMAL distribution    : {n_normal} / {n_total}")
print(f"Features with NON-NORMAL distribution: {n_total - n_normal} / {n_total}")


# ---------------------------------------------------------------
# 3. NORMALITY TEST FOR RESPONSE VARIABLE (blood_mg_dl)
# ---------------------------------------------------------------
x_tsb = df_clean[RESPONSE].dropna().values
stat_tsb, p_tsb = stats.kstest(x_tsb, 'norm', args=(x_tsb.mean(), x_tsb.std(ddof=1))) # type: ignore

tsb_row = {
    "feature": RESPONSE,
    "n": len(x_tsb),
    "mean": round(float(x_tsb.mean()), 4), # type: ignore
    "std": round(float(x_tsb.std(ddof=1)), 4), # type: ignore
    "skewness": round(float(stats.skew(x_tsb)), 4),
    "excess_kurtosis": round(float(stats.kurtosis(x_tsb)), 4),
    "KS_statistic": round(float(stat_tsb), 6),
    "KS_pvalue": round(float(p_tsb), 6),
    "normal_alpha_0.05": "Normal" if p_tsb > 0.05 else "Non-Normal",
}

print(f"\n=== Normality for {RESPONSE} ===")
print(f"KS Statistic : {stat_tsb:.4f}")
print(f"p-value      : {p_tsb:.2e}")
print(f"Conclusion   : {tsb_row['normal_alpha_0.05']}")


# ---------------------------------------------------------------
# 4. SAVE COMBINED CSV (features + TSB response)
# ---------------------------------------------------------------
tsb_df = pd.DataFrame([tsb_row])
combined_df = pd.concat([res_df, tsb_df], ignore_index=True)

import os
os.makedirs("__plots__/explore/csv", exist_ok=True)
combined_df.to_csv("__plots__/explore/csv/ks_test_results.csv", index=False)
print("\nFull table saved -> __plots__/explore/csv/ks_test_results.csv")
print(f"Total rows in CSV: {len(combined_df)} (45 features + 1 response variable)")
print(res_df.to_string(index=False))