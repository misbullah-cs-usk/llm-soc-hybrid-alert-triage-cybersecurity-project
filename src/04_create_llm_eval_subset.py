# src/04_create_llm_eval_subset.py

import os
import pandas as pd

TEST_PATH = "data/processed/test_processed.csv"
OUTPUT_PATH = "data/processed/llm_eval_subset.csv"

SAMPLE_SIZE = 300
RANDOM_STATE = 42
LABEL_COL = "label"


def main() -> None:
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv(TEST_PATH)

    print("Loaded test set")
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Required column '{LABEL_COL}' not found.")

    labels = sorted(df[LABEL_COL].dropna().unique())
    n_classes = len(labels)
    n_per_class = SAMPLE_SIZE // n_classes

    print("\nLabels found:", labels)
    print("Sampling per class:", n_per_class)

    sampled_parts = []

    for label in labels:
        class_df = df[df[LABEL_COL] == label]

        sampled = class_df.sample(
            n=min(n_per_class, len(class_df)),
            random_state=RANDOM_STATE
        )

        sampled_parts.append(sampled)

    subset = (
        pd.concat(sampled_parts, axis=0)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    # Keep the important columns explicitly, including label
    keep_cols = [
        "Id",
        "IncidentId",
        "AlertId",
        "Timestamp",
        "alert_text",
        "label"
    ]

    subset = subset[keep_cols]

    subset.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved LLM evaluation subset to: {OUTPUT_PATH}")
    print(f"Subset shape: {subset.shape}")

    print("\nSubset columns:")
    print(subset.columns.tolist())

    print("\nLabel distribution:")
    print(subset[LABEL_COL].value_counts())

    print("\nExample row:")
    print(subset.iloc[0].to_string())


if __name__ == "__main__":
    main()
