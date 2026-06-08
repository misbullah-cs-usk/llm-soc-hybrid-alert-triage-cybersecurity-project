# src/03_train_baseline.py

import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


TRAIN_PATH = "data/processed/train_processed.csv"
#TEST_PATH = "data/processed/test_processed.csv"
TEST_PATH = "data/processed/llm_eval_subset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_logreg_baseline.joblib")


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading processed data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    X_train = train_df["alert_text"].fillna("")
    y_train = train_df["label"]

    X_test = test_df["alert_text"].fillna("")
    y_test = test_df["label"]

    print("\nTraining TF-IDF + Logistic Regression baseline...")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix:")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print("Labels:", labels)
    print(cm)

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")

    results_df = test_df.copy()
    results_df["predicted_label"] = y_pred
    results_df.to_csv("data/processed/baseline_predictions.csv", index=False)

    print("Saved predictions to: data/processed/baseline_predictions.csv")


if __name__ == "__main__":
    main()
