# src/11_prepare_sft_dataset.py

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_PATH = "data/processed/train_processed.csv"
LLM_EVAL_PATH = "data/processed/llm_eval_subset.csv"

OUTPUT_DIR = "data/sft"
TRAIN_JSON = os.path.join(OUTPUT_DIR, "soc_triage_sft_train.json")
VAL_JSON = os.path.join(OUTPUT_DIR, "soc_triage_sft_val.json")
TEST_JSON = os.path.join(OUTPUT_DIR, "soc_triage_sft_test.json")

RANDOM_STATE = 42

# Start modestly. Increase later if training is stable.
MAX_TRAIN_ROWS = 12000
VAL_SIZE = 1000


SYSTEM_PROMPT = """You are a cybersecurity SOC alert triage analyst.
Classify the alert into exactly one label:
TruePositive, FalsePositive, or BenignPositive.
Return only valid JSON with this schema:
{"predicted_label":"TruePositive or FalsePositive or BenignPositive"}"""


def make_instruction(alert_text: str) -> str:
    return f"""Security alert:
{alert_text}

Return only JSON."""


def make_output(label: str) -> str:
    return json.dumps({"predicted_label": label}, ensure_ascii=False)


def to_alpaca_records(df: pd.DataFrame) -> list[dict]:
    records = []

    for _, row in df.iterrows():
        records.append({
            "instruction": SYSTEM_PROMPT,
            "input": make_instruction(row["alert_text"]),
            "output": make_output(row["label"])
        })

    return records


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    eval_df = pd.read_csv(LLM_EVAL_PATH)

    # Avoid leakage: remove rows used in your 300-row LLM evaluation subset
    eval_ids = set(eval_df["Id"].astype(str))
    train_df = train_df[~train_df["Id"].astype(str).isin(eval_ids)].copy()

    # Balanced training sample
    parts = []
    labels = sorted(train_df["label"].unique())
    per_class = MAX_TRAIN_ROWS // len(labels)

    for label in labels:
        class_df = train_df[train_df["label"] == label]
        n = min(per_class, len(class_df))
        parts.append(class_df.sample(n=n, random_state=RANDOM_STATE))

    sft_df = (
        pd.concat(parts)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    train_sft, val_sft = train_test_split(
        sft_df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=sft_df["label"]
    )

    train_records = to_alpaca_records(train_sft)
    val_records = to_alpaca_records(val_sft)
    test_records = to_alpaca_records(eval_df)

    with open(TRAIN_JSON, "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2, ensure_ascii=False)

    with open(VAL_JSON, "w", encoding="utf-8") as f:
        json.dump(val_records, f, indent=2, ensure_ascii=False)

    with open(TEST_JSON, "w", encoding="utf-8") as f:
        json.dump(test_records, f, indent=2, ensure_ascii=False)

    print(f"Saved train: {TRAIN_JSON} ({len(train_records)} records)")
    print(f"Saved val:   {VAL_JSON} ({len(val_records)} records)")
    print(f"Saved test:  {TEST_JSON} ({len(test_records)} records)")

    print("\nTraining label distribution:")
    print(train_sft["label"].value_counts())

    print("\nValidation label distribution:")
    print(val_sft["label"].value_counts())


if __name__ == "__main__":
    main()
