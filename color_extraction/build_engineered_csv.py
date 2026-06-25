"""
build_engineered_csv.py
Reads the existing training_engineered.csv and adds 11 engineered features,
producing training_engineered.csv in the same directory.

Does NOT touch images or re-run the extraction pipeline.
Runs on ALL rows (original + augmented) so the augmentation ratio is preserved.

Engineered features added (from Phase 4 keep list):
  Cross-zone means  (3):  mean_zones_Lab_b_mean, mean_zones_Cb_mean, mean_zones_B_mean
  R/B ratio         (3):  zone1_R_div_B, zone2_R_div_B, zone3_R_div_B
  G-B difference    (3):  zone1_G_minus_B, zone2_G_minus_B, zone3_G_minus_B
  Cross-zone mean   (1):  mean_zones_S_mean
  Log postnatal age (1):  log1p_postnatal_age_days

  (mean_zones_H_mean, all gradients, all ITA dropped per Phase 4 results)

Output column order:
  patient_id, is_augmented
  zone1_* (14), zone2_* (14), zone3_* (14)         — original 42
  gestational_age, postnatal_age_days, weight        — original meta
  [11 engineered features]
  blood_mg_dl, jaundice_label                        — labels last

Usage:
  python build_engineered_csv.py
  python build_engineered_csv.py --input path/to/training_engineered.csv --output path/to/out.csv
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("engineer")

DEFAULT_INPUT  = "__data__/neo/out/training_engineered.csv"
DEFAULT_OUTPUT = "__data__/neo/out/training_engineered.csv"

ZONES         = ["zone1", "zone2", "zone3"]
FEATURE_NAMES = [
    "R_mean", "G_mean", "B_mean", "R_std",
    "Y_mean", "Cr_mean", "Cb_mean",
    "H_mean", "S_mean", "L_mean",
    "Lab_L_mean", "Lab_a_mean", "Lab_b_mean", "Lab_L_std",
]

# Exact ordered output for engineered block
ENGINEERED_NAMES = [
    "mean_zones_Lab_b_mean",
    "mean_zones_Cb_mean",
    "mean_zones_B_mean",
    "mean_zones_S_mean",
    "zone1_R_div_B",
    "zone2_R_div_B",
    "zone3_R_div_B",
    "zone1_G_minus_B",
    "zone2_G_minus_B",
    "zone3_G_minus_B",
    "log1p_postnatal_age_days",
]


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise division; sets NaN where denominator is zero."""
    result = num.copy().astype(float)
    zero   = den == 0
    result[zero]  = np.nan
    result[~zero] = num[~zero] / den[~zero]
    return result


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Cross-zone means ──────────────────────────────────────────────────────
    for ch in ["Lab_b_mean", "Cb_mean", "B_mean", "S_mean"]:
        cols = [f"{z}_{ch}" for z in ZONES]
        df[f"mean_zones_{ch}"] = df[cols].mean(axis=1)
        log.info("  mean_zones_%-14s  non-null: %d", ch, df[f"mean_zones_{ch}"].notna().sum())

    # ── R/B ratio per zone ───────────────────────────────────────────────────
    for zone in ZONES:
        r_col = f"{zone}_R_mean"
        b_col = f"{zone}_B_mean"
        df[f"{zone}_R_div_B"] = _safe_div(df[r_col], df[b_col])
        log.info("  %-22s  non-null: %d", f"{zone}_R_div_B", df[f"{zone}_R_div_B"].notna().sum())

    # ── G-B difference per zone ───────────────────────────────────────────────
    for zone in ZONES:
        g_col = f"{zone}_G_mean"
        b_col = f"{zone}_B_mean"
        df[f"{zone}_G_minus_B"] = df[g_col] - df[b_col]
        log.info("  %-22s  non-null: %d", f"{zone}_G_minus_B", df[f"{zone}_G_minus_B"].notna().sum())

    # ── Log postnatal age ─────────────────────────────────────────────────────
    df["log1p_postnatal_age_days"] = np.log1p(df["postnatal_age_days"])
    log.info("  %-22s  non-null: %d", "log1p_postnatal_age_days",
             df["log1p_postnatal_age_days"].notna().sum())

    return df


def build_column_order(df: pd.DataFrame) -> list[str]:
    """
    Enforce deterministic column order:
      patient_id, is_augmented
      42 original color features (zone1 × 14, zone2 × 14, zone3 × 14)
      gestational_age, postnatal_age_days, weight
      11 engineered features
      blood_mg_dl, jaundice_label
    """
    zone_cols = [f"{z}_{f}" for z in ZONES for f in FEATURE_NAMES]
    meta_cols = ["gestational_age", "postnatal_age_days", "weight"]
    label_cols = ["blood_mg_dl", "jaundice_label"]

    ordered = (
        ["patient_id", "is_augmented"]
        + zone_cols
        + meta_cols
        + ENGINEERED_NAMES
        + label_cols
    )
    # safety: keep any unexpected columns at the end rather than silently drop them
    extras = [c for c in df.columns if c not in ordered]
    if extras:
        log.warning("unexpected columns appended at end: %s", extras)
    return [c for c in ordered + extras if c in df.columns]


def main(input_path: str, output_path: str) -> None:
    log.info("reading  %s", input_path)
    df = pd.read_csv(input_path)
    log.info("loaded   %d rows × %d cols  (orig=%d  aug=%d)",
             len(df), len(df.columns),
             (~df["is_augmented"]).sum(), df["is_augmented"].sum())

    log.info("engineering features …")
    df = add_engineered_features(df)

    col_order = build_column_order(df)
    df = df[col_order]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("saved    %s  (%d rows × %d cols)", output_path, *df.shape)

    # ── Quick sanity check ────────────────────────────────────────────────────
    log.info("── sanity check ─────────────────────────────────────────────────")
    orig = df[~df["is_augmented"]]
    for name in ENGINEERED_NAMES:
        if name not in df.columns:
            log.error("  MISSING: %s", name)
            continue
        null_n = orig[name].isna().sum()
        log.info("  %-30s  orig_nulls=%d  orig_mean=%.4f",
                 name, null_n, orig[name].mean())

    log.info("done. total columns: %d  (42 original + 3 meta + 11 engineered + 2 labels + 2 id)",
             len(df.columns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add engineered features to training_engineered.csv")
    parser.add_argument("--input",  default=DEFAULT_INPUT,  help="Path to training_engineered.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path for output CSV")
    args = parser.parse_args()
    main(args.input, args.output)