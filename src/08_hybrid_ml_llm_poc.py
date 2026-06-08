# src/08_hybrid_ml_llm_poc.py

import json
import re
import time
import joblib
import requests
import pandas as pd
from tqdm import tqdm


INPUT_PATH = "data/processed/llm_eval_subset.csv"
MODEL_PATH = "models/tfidf_logreg_baseline.joblib"
OUTPUT_PATH = "data/processed/hybrid_ml_llm_outputs.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b-instruct"

MAX_ROWS = None


def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 4096
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )

    response.raise_for_status()
    return response.json()["response"]


def extract_json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "analyst_summary": "Could not parse JSON output.",
        "risk_explanation": "",
        "recommended_actions": [],
        "mitre_interpretation": [],
        "containment_priority": "Unknown"
    }


def build_hybrid_prompt(alert_text: str, ml_prediction: str) -> str:
    return f"""
You are a cybersecurity SOC analyst.

A machine learning classifier has already predicted this alert as:

{ml_prediction}

Your job is NOT to change the label.
Your job is to generate an analyst-facing explanation and response guidance.

Return ONLY valid JSON.
Do not include markdown.
Do not include text outside the JSON.

Required JSON schema:
{{
  "analyst_summary": "brief plain-English summary of the alert",
  "risk_explanation": "why this alert may matter in a SOC context",
  "mitre_interpretation": [
    "possible MITRE ATT&CK tactic or technique based on the alert fields"
  ],
  "recommended_actions": [
    "specific action 1",
    "specific action 2",
    "specific action 3",
    "specific action 4"
  ],
  "containment_priority": "Low or Medium or High"
}}

Alert:
{alert_text}
""".strip()


def main():
    df = pd.read_csv(INPUT_PATH)

    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS).copy()

    classifier = joblib.load(MODEL_PATH)

    ml_predictions = classifier.predict(df["alert_text"].fillna(""))

    results = []

    print(f"Running hybrid ML + Qwen PoC on {len(df)} rows...")
    print(f"Using LLM model: {MODEL_NAME}")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        alert_text = row["alert_text"]
        true_label = row["label"]
        ml_prediction = ml_predictions[i]

        prompt = build_hybrid_prompt(alert_text, ml_prediction)

        start = time.time()

        try:
            raw_response = call_ollama(prompt)
            parsed = extract_json(raw_response)
        except Exception as e:
            raw_response = str(e)
            parsed = {
                "analyst_summary": f"Ollama request failed: {e}",
                "risk_explanation": "",
                "mitre_interpretation": [],
                "recommended_actions": [],
                "containment_priority": "Unknown"
            }

        latency = time.time() - start

        results.append({
            "Id": row["Id"],
            "IncidentId": row["IncidentId"],
            "AlertId": row["AlertId"],
            "Timestamp": row["Timestamp"],
            "alert_text": alert_text,
            "true_label": true_label,
            "ml_predicted_label": ml_prediction,
            "is_correct": true_label == ml_prediction,
            "analyst_summary": parsed.get("analyst_summary", ""),
            "risk_explanation": parsed.get("risk_explanation", ""),
            "mitre_interpretation": json.dumps(parsed.get("mitre_interpretation", [])),
            "recommended_actions": json.dumps(parsed.get("recommended_actions", [])),
            "containment_priority": parsed.get("containment_priority", "Unknown"),
            "raw_llm_response": raw_response,
            "latency_seconds": latency
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_PATH, index=False)

    accuracy = output_df["is_correct"].mean()

    print(f"\nSaved hybrid PoC output to: {OUTPUT_PATH}")
    print(f"Hybrid prediction accuracy: {accuracy:.4f}")
    print(f"Average LLM explanation latency: {output_df['latency_seconds'].mean():.2f} seconds")

    print("\nExample hybrid output:")
    example = output_df.iloc[0]
    print("True label:", example["true_label"])
    print("ML prediction:", example["ml_predicted_label"])
    print("Summary:", example["analyst_summary"])
    print("Actions:", example["recommended_actions"])


if __name__ == "__main__":
    main()
