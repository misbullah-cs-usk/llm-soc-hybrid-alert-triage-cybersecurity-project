# src/02_preprocess.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_PATH = "data/raw/GUIDE_Train.csv"
OUTPUT_DIR = "data/processed"

TARGET_COL = "IncidentGrade"

FEATURE_COLUMNS = [
    "Category",
    "MitreTechniques",
    "EntityType",
    "EvidenceRole",
    "DetectorId",
    "AlertTitle",
    "SuspicionLevel",
    "LastVerdict",
    "CountryCode",
    "State",
    "City",
    "OSFamily",
    "OSVersion",
]


def build_text_feature(row: pd.Series) -> str:
    """
    Convert one alert row into a text-like record.
    This is useful for TF-IDF, classical ML, and later LLM prompting.
    """
    parts = []

    for col in FEATURE_COLUMNS:
        value = row.get(col, "")
        if pd.isna(value):
            value = "unknown"
        parts.append(f"{col}: {value}")

    return " | ".join(parts)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading dataset from {RAW_PATH}...")

    # Start with a sample so your computer does not crash.
    # You can increase this later.
    df = pd.read_csv(RAW_PATH, nrows=200_000)

    print(f"Raw shape: {df.shape}")

    # Keep rows with valid labels
    df = df.dropna(subset=[TARGET_COL])

    print("\nLabel distribution:")
    print(df[TARGET_COL].value_counts())

    # Build combined text field
    df["alert_text"] = df.apply(build_text_feature, axis=1)

    processed = df[["Id", "IncidentId", "AlertId", "Timestamp", "alert_text", TARGET_COL]].copy()

    processed = processed.rename(columns={TARGET_COL: "label"})

    train_df, test_df = train_test_split(
        processed,
        test_size=0.2,
        random_state=42,
        stratify=processed["label"]
    )

    train_path = os.path.join(OUTPUT_DIR, "train_processed.csv")
    test_path = os.path.join(OUTPUT_DIR, "test_processed.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved train set to: {train_path}")
    print(f"Saved test set to: {test_path}")

    print("\nExample processed row:")
    print(train_df.iloc[0]["alert_text"])
    print("Label:", train_df.iloc[0]["label"])


if __name__ == "__main__":
    main()
