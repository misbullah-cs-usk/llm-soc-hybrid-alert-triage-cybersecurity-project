# src/03a_rule_based_baseline.py

import os
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


TEST_PATH = "data/processed/test_processed.csv"
OUTPUT_PATH = "data/processed/rule_based_predictions.csv"


def predict_rule_based(alert_text: str) -> str:
    """
    Simple SOC-style heuristic baseline.

    Labels:
    - TruePositive: likely real malicious incident
    - FalsePositive: likely incorrectly triggered alert
    - BenignPositive: alert is technically valid but benign/expected
    """

    text = str(alert_text).lower()

    # Strong malicious indicators
    true_positive_keywords = [
        "credentialaccess",
        "commandandcontrol",
        "exfiltration",
        "lateralmovement",
        "execution",
        "persistence",
        "privilegeescalation",
        "defenseevasion",
        "t1110",
        "t1078",
        "t1021",
        "t1047",
        "t1105",
        "t1569",
        "suspicionlevel: suspicious",
        "lastverdict: malicious",
    ]

    # Likely benign or low-risk indicators
    benign_keywords = [
        "benign",
        "lastverdict: clean",
        "lastverdict: benign",
        "suspicionlevel: unknown",
        "category: discovery",
        "category: initialaccess",
    ]

    # Likely noisy / insufficient evidence indicators
    false_positive_keywords = [
        "falsepositive",
        "lastverdict: unknown",
        "mitretechniques: unknown",
        "entitytype: url",
        "evidencerole: related",
    ]

    tp_score = sum(1 for kw in true_positive_keywords if kw in text)
    benign_score = sum(1 for kw in benign_keywords if kw in text)
    fp_score = sum(1 for kw in false_positive_keywords if kw in text)

    # Decision logic
    if tp_score >= 2:
        return "TruePositive"

    if fp_score >= 2 and tp_score == 0:
        return "FalsePositive"

    if benign_score >= 1 and tp_score == 0:
        return "BenignPositive"

    # Default majority-class fallback from your sample distribution
    return "BenignPositive"


def main() -> None:
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"{TEST_PATH} not found. Run src/02_preprocess.py first."
        )

    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)

    y_true = test_df["label"]
    y_pred = test_df["alert_text"].apply(predict_rule_based)

    acc = accuracy_score(y_true, y_pred)

    print("\nRule-Based Baseline Results")
    print("=" * 80)
    print(f"Accuracy: {acc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion matrix:")
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Labels:", labels)
    print(cm)

    output_df = test_df.copy()
    output_df["predicted_label"] = y_pred
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
