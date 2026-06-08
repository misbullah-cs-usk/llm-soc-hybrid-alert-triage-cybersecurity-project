# src/01_explore_dataset.py

import os
import pandas as pd


DATA_DIR = "data/raw"


def inspect_csv(file_path: str, nrows: int = 5) -> None:
    print("=" * 100)
    print(f"File: {file_path}")

    try:
        df = pd.read_csv(file_path, nrows=nrows)
        print(f"Shape preview: {df.shape}")
        print("\nColumns:")
        for col in df.columns:
            print(f"  - {col}")

        print("\nFirst rows:")
        print(df.head(nrows).to_string())

    except Exception as e:
        print(f"Could not read file: {e}")


def main() -> None:
    if not os.path.exists(DATA_DIR):
        print(f"Folder not found: {DATA_DIR}")
        print("Create data/raw and put your Kaggle CSV files there.")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

    if not files:
        print("No CSV files found in data/raw.")
        return

    print(f"Found {len(files)} CSV file(s):")
    for file in files:
        print(f"  - {file}")

    for file in files:
        inspect_csv(os.path.join(DATA_DIR, file))


if __name__ == "__main__":
    main()
