import pickle
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATASETS_DIR = REPO_ROOT / "datasets"
PROJECT_DIR = REPO_ROOT / "project"

utils_path = PROJECT_DIR / "utils.py"

if not utils_path.exists():
    raise FileNotFoundError(
        f"utils.py not found at: {utils_path}"
    )

sys.path.insert(0, str(PROJECT_DIR))

from utils import clean_nested_columns

input_path = DATASETS_DIR / "LSWMD.pkl"
output_path = DATASETS_DIR / "Dataset.pkl"

if not input_path.exists():
    raise FileNotFoundError(
        f"Raw dataset not found at: {input_path}"
    )

file_size_mb = input_path.stat().st_size / (1024 * 1024)
print(f"Original file size: {file_size_mb:.2f} MB")

# Compatibility patch for old pandas pickle files
sys.modules["pandas.indexes"] = pd.core.indexes
sys.modules["pandas.indexes.base"] = pd.core.indexes.base

with input_path.open("rb") as file:
    df = pickle.load(file, encoding="latin1")

print("Dataset successfully loaded.")
print(f"Original rows: {len(df):,}")

df = clean_nested_columns(df)

df = df[
    df["trainTestLabel"].isin(["training", "test"])
].reset_index(drop=True)

print(f"Remaining rows after filtering: {len(df):,}")
print(df.head())

df.to_pickle(output_path)

print(f"Clean dataset saved to: {output_path}")