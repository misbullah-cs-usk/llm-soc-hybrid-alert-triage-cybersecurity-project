# src/04b_evaluate_baselines_on_llm_subset.py

import os
import sys
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Allow importing helper file from src/
sys.path.append("src")

from rule_based_prediction import predict_rule_based


SUBSET_PATH = "data/processed/llm_eval_subset.csv"
LOGREG_MODEL_PATH = "models/tfidf_logreg_baseline.joblib"

OUTPUT_PREDICTIONS_PATH = "data/processed/subset_baseline_predictions.csv"
OUTPUT_SUMMARY_PATH = "reports/subset_baseline_results.csv"

LABELS = [
    "BenignPositive",
    "FalsePositive",
    "TruePositive",
]


def evaluate_model(name: str, y_true, y_pred) -> dict:
    accuracy = accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    ))

    print("\nConfusion matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    print("Labels:", LABELS)
    print(cm)

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
    }


def main() -> None:
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    if not os.path.exists(SUBSET_PATH):
        raise FileNotFoundError(
            f"{SUBSET_PATH} not found. Run src/04_create_llm_eval_subset.py first."
        )

    if not os.path.exists(LOGREG_MODEL_PATH):
        raise FileNotFoundError(
            f"{LOGREG_MODEL_PATH} not found. Run src/03_train_baseline.py first."
        )

    df = pd.read_csv(SUBSET_PATH)

    print("Loaded LLM evaluation subset")
    print("Shape:", df.shape)
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    y_true = df["label"]

    # 1. Rule-based baseline
    rule_pred = df["alert_text"].fillna("").apply(predict_rule_based)

    # 2. Logistic Regression baseline
    logreg_model = joblib.load(LOGREG_MODEL_PATH)
    logreg_pred = logreg_model.predict(df["alert_text"].fillna(""))

    results = []
    results.append(evaluate_model("Rule-Based Baseline", y_true, rule_pred))
    results.append(evaluate_model("TF-IDF + Logistic Regression", y_true, logreg_pred))

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    prediction_df = df.copy()
    prediction_df["rule_based_prediction"] = rule_pred
    prediction_df["logreg_prediction"] = logreg_pred
    prediction_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)

    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    print(f"\nSaved summary to: {OUTPUT_SUMMARY_PATH}")
    print(f"Saved predictions to: {OUTPUT_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
