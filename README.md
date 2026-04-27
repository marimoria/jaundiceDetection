# Jaundice Detection

Description to edit later

## DHS Data Manager

This project extracts specific variables from DHS (Demographic and Health Surveys) datasets across 5 countries and combines them into a single CSV file for analysis.

> ⚠️ **Important:** DHS data is licensed. Do not share raw data publicly. Your coworker must have their own approved DHS account. Keep the HuggingFace repository **private** at all times.

### Countries Covered

| Country     | Code |
| ----------- | ---- |
| Bangladesh  | BD   |
| India       | IA   |
| Indonesia   | ID   |
| Nepal       | NP   |
| Timor-Leste | TL   |

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/jaundiceDetection
cd jaundiceDetection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Create your `.env` file

Create a file called `.env` in the root of the project.

```
HF_TOKEN=hf_your_hunggingface_write_access_token_here
DATA_DIR=D:/Your/Path/To/__data__
```

- `HF_TOKEN` — Your HuggingFace access token. Get it at https://huggingface.co/settings/tokens
- `DATA_DIR` — Path to your local data folder where DHS files are stored and CSV will be saved

> Your coworker must be added as a collaborator on the HuggingFace dataset repo before their token will work. Contact the repo owner to be added.

### 4. Run the Manager

```bash
python dhs_manager.py
```

## DHS Manager Menu

```
========================================
       DHS DATA MANAGER
========================================
1. Download full DHS data from HuggingFace
2. Extract variables from DTA → save CSV
3. Upload combined CSV to HuggingFace
4. Download combined CSV from HuggingFace
0. Exit
========================================
```

| Option | What It Does                                                                         |
| ------ | ------------------------------------------------------------------------------------ |
| 1      | Downloads the full raw DHS dataset (~15GB) from HuggingFace to your local `DATA_DIR` |
| 2      | Reads the raw DTA files and extracts specific variables into `dhs_combined.csv`      |
| 3      | Uploads the combined CSV to HuggingFace private repo                                 |
| 4      | Downloads just the combined CSV from HuggingFace — no need to download full 15GB     |

## Extracted Variables

### KR File — Children's Recode

| Variable | Label                                  |
| -------- | -------------------------------------- |
| `m18`    | Size of child at birth                 |
| `m19`    | Birth weight in kilograms (3 decimals) |
| `m19a`   | Weight at birth/recall                 |
| `m17`    | Delivery by caesarean section          |
| `b4`     | Sex of child                           |
| `m4`     | Duration of breastfeeding              |
| `m5`     | Months of breastfeeding                |

### IR File — Individual Recode (Women)

| Variable | Label                                                                 |
| -------- | --------------------------------------------------------------------- |
| `v453`   | Hemoglobin level (g/dl - 1 decimal)                                   |
| `v454`   | Currently pregnant (from household questionnaire)                     |
| `v455`   | Result of measurement - hemoglobin                                    |
| `v456`   | Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal) |
| `v457`   | Anemia level                                                          |

### BR File — Birth Recode

| Variable | Label                              |
| -------- | ---------------------------------- |
| `b11`    | Preceding birth interval (months)  |
| `b12`    | Succeeding birth interval (months) |

## CSV Structure

The output file `dhs_combined.csv` has the following structure:

| Column         | Description                              |
| -------------- | ---------------------------------------- |
| `country`      | Full country name (e.g. Bangladesh)      |
| `country_code` | 2-letter DHS country code (e.g. BD)      |
| `file_type`    | Source file type: `KR`, `IR`, or `BR`    |
| `m17`          | Caesarean section                        |
| `m18`          | Size of child at birth                   |
| `m19`          | Birth weight (kg)                        |
| `m19a`         | Weight at birth/recall                   |
| `b4`           | Sex of child                             |
| `m4`           | Duration of breastfeeding                |
| `m5`           | Months of breastfeeding                  |
| `v453`         | Hemoglobin level                         |
| `v454`         | Currently pregnant                       |
| `v455`         | Hemoglobin measurement result            |
| `v456`         | Hemoglobin adjusted for altitude/smoking |
| `v457`         | Anemia level                             |
| `b11`          | Preceding birth interval (months)        |
| `b12`          | Succeeding birth interval (months)       |

> Columns not applicable to a file type will be empty (e.g. KR rows will have empty `v453`–`v457` columns).

### Example rows:

| country    | country_code | file_type | m18 | m19   | b4  | v453 | b11 |
| ---------- | ------------ | --------- | --- | ----- | --- | ---- | --- |
| Bangladesh | BD           | KR        | 3   | 2.800 | 1   |      |     |
| Bangladesh | BD           | IR        |     |       |     | 128  |     |
| Bangladesh | BD           | BR        |     |       |     |      | 24  |
| India      | IA           | KR        | 2   | 3.100 | 2   |      |     |

## File Naming Convention

DHS files follow a strict naming pattern:

```
B  D  B  R  8  1  D  T
│  │  │  │  │  │  │  │
│  │  │  │  │  │  └──└── DT = folder suffix
│  │  │  │  └──└──────── 81 = survey round
│  │  └──└────────────── BR = file type
└──└──────────────────── BD = country code
```

Files inside the folder follow the same pattern but end in `FL`:

```
BDBR81DT/
└── BDBR81FL.DTA   ← actual data file
└── BDBR81FL.DO    ← variable documentation
└── BDBR81FL.MAP   ← metadata
```

## Project Structure

```
jaundiceDetection/
├── dhs_manager.py        ← main script
├── requirements.txt      ← dependencies
├── .env                  ← your tokens and paths (never push to GitHub!)
├── .gitignore            ← must include .env and __data__/
└── __data__/
    ├── dhs_combined.csv  ← extracted CSV output
    ├── Bangladesh/
    │   ├── BDBR81DT/
    │   ├── BDKR81DT/
    │   └── BDIR81DT/
    ├── India/
    ├── Indonesia/
    ├── Nepal/
    └── TimorLeste/
```
