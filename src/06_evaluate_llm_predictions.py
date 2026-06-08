# src/06_evaluate_llm_predictions.py

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PRED_PATH = "data/processed/llm_predictions_ollama.csv"
#PRED_PATH = "data/processed/llm_predictions_fewshot_ollama.csv"

VALID_LABELS = [
    "BenignPositive",
    "FalsePositive",
    "TruePositive"
]


def main() -> None:
    df = pd.read_csv(PRED_PATH)

    y_true = df["true_label"]
    y_pred = df["llm_predicted_label"]

    print("LLM Evaluation")
    print("=" * 80)

    strict_accuracy = accuracy_score(y_true, y_pred)
    print(f"Strict accuracy including invalid labels: {strict_accuracy:.4f}")

    valid_df = df[df["llm_predicted_label"].isin(VALID_LABELS)].copy()

    print(f"\nTotal rows: {len(df)}")
    print(f"Valid LLM predictions: {len(valid_df)}")
    print(f"Invalid predictions: {len(df) - len(valid_df)}")

    if len(valid_df) > 0:
        print("\nAccuracy on valid predictions only:")
        print(accuracy_score(
            valid_df["true_label"],
            valid_df["llm_predicted_label"]
        ))

        print("\nClassification report:")
        print(classification_report(
            valid_df["true_label"],
            valid_df["llm_predicted_label"],
            labels=VALID_LABELS
        ))

        print("\nConfusion matrix:")
        cm = confusion_matrix(
            valid_df["true_label"],
            valid_df["llm_predicted_label"],
            labels=VALID_LABELS
        )
        print("Labels:", VALID_LABELS)
        print(cm)

    print("\nAverage latency:")
    print(df["latency_seconds"].mean())


if __name__ == "__main__":
    main()
