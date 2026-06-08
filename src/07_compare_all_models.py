# src/07_compare_all_models.py

import os
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from rule_based_prediction import predict_rule_based


SUBSET_PATH = "data/processed/llm_eval_subset.csv"
LOGREG_MODEL_PATH = "models/tfidf_logreg_baseline.joblib"

ZERO_SHOT_PATH = "data/processed/llm_predictions_ollama.csv"
FEW_SHOT_PATH = "data/processed/llm_predictions_fewshot_ollama.csv"

OUTPUT_PATH = "reports/all_model_comparison.csv"

VALID_LABELS = ["BenignPositive", "FalsePositive", "TruePositive"]


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=VALID_LABELS,
        average="macro",
        zero_division=0
    )

    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=VALID_LABELS,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_f1": weighted_f1
    }


def add_result(results, model_name, role, y_true, y_pred):
    metrics = compute_metrics(y_true, y_pred)

    results.append({
        "model": model_name,
        "role": role,
        "accuracy": round(metrics["accuracy"], 4),
        "macro_precision": round(metrics["macro_precision"], 4),
        "macro_recall": round(metrics["macro_recall"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "weighted_f1": round(metrics["weighted_f1"], 4)
    })


def main():
    os.makedirs("reports", exist_ok=True)

    subset = pd.read_csv(SUBSET_PATH)
    y_true = subset["label"]

    results = []

    # Rule-based baseline
    rule_pred = subset["alert_text"].apply(predict_rule_based)
    add_result(
        results,
        "Rule-Based Baseline",
        "Traditional heuristic SOC logic",
        y_true,
        rule_pred
    )

    # Logistic Regression baseline
    logreg_model = joblib.load(LOGREG_MODEL_PATH)
    logreg_pred = logreg_model.predict(subset["alert_text"].fillna(""))
    add_result(
        results,
        "TF-IDF + Logistic Regression",
        "Classical ML classifier",
        y_true,
        logreg_pred
    )

    # Qwen zero-shot
    zero_df = pd.read_csv(ZERO_SHOT_PATH)
    add_result(
        results,
        "Qwen Zero-Shot",
        "Direct LLM classification",
        zero_df["true_label"],
        zero_df["llm_predicted_label"]
    )

    # Qwen few-shot
    few_df = pd.read_csv(FEW_SHOT_PATH)
    add_result(
        results,
        "Qwen Few-Shot",
        "Prompt-engineered LLM classification",
        few_df["true_label"],
        few_df["llm_predicted_label"]
    )

    # Hybrid system accuracy equals Logistic Regression prediction accuracy
    add_result(
        results,
        "Hybrid ML + Qwen",
        "ML prediction with LLM explanation and playbook generation",
        y_true,
        logreg_pred
    )

    comparison = pd.DataFrame(results)
    comparison.to_csv(OUTPUT_PATH, index=False)

    print(comparison.to_string(index=False))
    print(f"\nSaved comparison table to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
