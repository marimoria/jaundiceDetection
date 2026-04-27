# Convert the downloaded dataset into 1 csv file.
# Since the dataset can only be accessed with request, the csv file must not be hosted on GitHub.
# Please use this once after downloading the dhs dataset and then you may remove the uncessary files in __dhs__ folder.

import os
import glob
import pandas as pd
import pyreadstat
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATA_DIR = "D:/Code/jaundiceDetection/__data__/__dhs__"
OUTPUT_CSV = "dhs_combined.csv"

COUNTRIES = {
    "Bangladesh": "BD",
    "India": "IA",
    "Indonesia": "ID",
    "Nepal": "NP",
    "TimorLeste": "TL",
}

# Variables to extract per file type
FILE_VARIABLES = {
    "KR": ["m18", "m19", "m19a", "m17", "b4", "m4", "m5"],
    "IR": ["v453", "v454", "v455", "v456", "v457"],
    "BR": ["b11", "b12"],
}


# ─────────────────────────────────────────
# HELPER: Find DTA file for a country + file type
# ─────────────────────────────────────────
def find_dta(country_dir, country_code, file_type):
    """
    Searches for pattern like: BD/BDBR81DT/BDBR81FL.DTA
    """
    pattern = os.path.join(
        country_dir,
        f"{country_code}{file_type}*DT",
        f"{country_code}{file_type}*FL.DTA",
    )
    matches = glob.glob(pattern, recursive=False)
    if not matches:
        print(f"  ⚠ Not found: {country_code} {file_type} → {pattern}")
        return None
    return matches[0]


# ─────────────────────────────────────────
# HELPER: Read DTA and extract columns safely
# ─────────────────────────────────────────
def read_dta(filepath, variables):
    try:
        df, meta = pyreadstat.read_dta(filepath, usecols=variables)  # type: ignore
        existing = [v for v in variables if v in df.columns]  # type: ignore
        missing = [v for v in variables if v not in df.columns]  # type: ignore
        if missing:
            print(f"Missing columns: {missing}")
        return df[existing]  # type: ignore
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
all_dfs = []

for country_name, country_code in COUNTRIES.items():
    print(f"\n{'='*40}")
    print(f"Processing: {country_name} ({country_code})")
    print(f"{'='*40}")

    country_dir = os.path.join(DATA_DIR, country_name)
    country_dfs = []

    for file_type, variables in FILE_VARIABLES.items():
        print(f"\n  [{file_type}] Looking for variables: {variables}")

        dta_path = find_dta(country_dir, country_code, file_type)
        if not dta_path:
            continue

        print(f"  ✓ Found: {dta_path}")
        df = read_dta(dta_path, variables)
        if df is None or df.empty:
            continue

        print(f"  ✓ Rows: {len(df):,} | Columns: {list(df.columns)}")
        df["file_type"] = file_type
        country_dfs.append(df)

    if not country_dfs:
        print(f"  ✗ No data found for {country_name}")
        continue

    # Combine all file types for this country
    country_combined = pd.concat(country_dfs, axis=0, ignore_index=True)
    country_combined.insert(0, "country", country_name)
    country_combined.insert(1, "country_code", country_code)

    all_dfs.append(country_combined)
    print(f"\n  ✓ {country_name} total rows: {len(country_combined):,}")


# ─────────────────────────────────────────
# COMBINE ALL COUNTRIES & SAVE
# ─────────────────────────────────────────
if not all_dfs:
    print("\n✗ No data extracted. Check your file paths.")
else:
    print(f"\n{'='*40}")
    print("Combining all countries...")
    final_df = pd.concat(all_dfs, axis=0, ignore_index=True)

    # Add a separator column so country boundaries are clear
    final_df.insert(2, "separator", "")

    print(f"✓ Total rows: {len(final_df):,}")
    print(f"✓ Columns: {list(final_df.columns)}")

    # Save
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved to: {OUTPUT_CSV}")

    # Summary per country
    print(f"\n{'='*40}")
    print("Summary:")
    print(
        final_df.groupby(["country", "file_type"])
        .size()
        .reset_index(name="rows")
        .to_string(index=False)
    )
