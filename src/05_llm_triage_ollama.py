# src/05_llm_triage_ollama.py

import json
import re
import time
import requests
import pandas as pd
from tqdm import tqdm


INPUT_PATH = "data/processed/llm_eval_subset.csv"
OUTPUT_PATH = "data/processed/llm_predictions_ollama.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"

# Change this if needed:
#MODEL_NAME = "qwen2.5:7b-instruct"
MODEL_NAME = "qwen2.5:3b-instruct"
# MODEL_NAME = "qwen3:4b"

# Start small first. Set to None later for all 300 rows.
MAX_ROWS = None


VALID_LABELS = {
    "TruePositive",
    "FalsePositive",
    "BenignPositive"
}


def build_prompt(alert_text: str) -> str:
    return f"""
You are a cybersecurity SOC alert triage analyst.

Your task is to classify the following security alert into exactly one of these labels:

1. TruePositive
Meaning: the alert likely represents real malicious or suspicious security activity.

2. FalsePositive
Meaning: the alert was likely triggered incorrectly and does not represent a real security issue.

3. BenignPositive
Meaning: the alert is technically valid, but the activity is likely benign, expected, or low-risk.

Return ONLY valid JSON.
Do not include markdown.
Do not include explanation outside the JSON.

Required JSON schema:
{{
  "predicted_label": "TruePositive or FalsePositive or BenignPositive",
  "confidence": 0.0,
  "reasoning_summary": "brief analyst explanation",
  "recommended_actions": [
    "action 1",
    "action 2",
    "action 3"
  ]
}}

Security alert:
{alert_text}
""".strip()


def call_ollama(prompt: str, model_name: str = MODEL_NAME) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
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
    """
    Tries to extract valid JSON from the model output.
    Handles cases where the model accidentally adds extra text.
    """
    text = text.strip()

    # Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract first JSON object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "predicted_label": "PARSE_ERROR",
        "confidence": 0.0,
        "reasoning_summary": "Could not parse model output as JSON.",
        "recommended_actions": []
    }


def normalize_label(label: str) -> str:
    label = str(label).strip()

    mapping = {
        "true positive": "TruePositive",
        "truepositive": "TruePositive",
        "malicious": "TruePositive",
        "real threat": "TruePositive",

        "false positive": "FalsePositive",
        "falsepositive": "FalsePositive",

        "benign positive": "BenignPositive",
        "benignpositive": "BenignPositive",
        "benign": "BenignPositive"
    }

    key = label.lower().replace("_", " ").strip()

    if label in VALID_LABELS:
        return label

    if key in mapping:
        return mapping[key]

    compact_key = key.replace(" ", "")
    if compact_key in mapping:
        return mapping[compact_key]

    return "INVALID_LABEL"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS).copy()

    results = []

    print(f"Using model: {MODEL_NAME}")
    print(f"Input rows: {len(df)}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        alert_text = row["alert_text"]
        true_label = row["label"]

        prompt = build_prompt(alert_text)

        start_time = time.time()

        try:
            raw_response = call_ollama(prompt)
            parsed = extract_json(raw_response)
        except Exception as e:
            raw_response = str(e)
            parsed = {
                "predicted_label": "REQUEST_ERROR",
                "confidence": 0.0,
                "reasoning_summary": f"Ollama request failed: {e}",
                "recommended_actions": []
            }

        latency = time.time() - start_time

        predicted_label_raw = parsed.get("predicted_label", "INVALID_LABEL")
        predicted_label = normalize_label(predicted_label_raw)

        results.append({
            "Id": row["Id"],
            "IncidentId": row["IncidentId"],
            "AlertId": row["AlertId"],
            "Timestamp": row["Timestamp"],
            "alert_text": alert_text,
            "true_label": true_label,
            "llm_predicted_label_raw": predicted_label_raw,
            "llm_predicted_label": predicted_label,
            "llm_confidence": parsed.get("confidence", 0.0),
            "llm_reasoning_summary": parsed.get("reasoning_summary", ""),
            "llm_recommended_actions": json.dumps(parsed.get("recommended_actions", [])),
            "raw_response": raw_response,
            "latency_seconds": latency
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved LLM predictions to: {OUTPUT_PATH}")

    valid_df = output_df[
        output_df["llm_predicted_label"].isin(VALID_LABELS)
    ].copy()

    total = len(output_df)
    valid = len(valid_df)
    invalid = total - valid

    print("\nPrediction validity:")
    print(f"Total rows: {total}")
    print(f"Valid predictions: {valid}")
    print(f"Invalid / parse errors: {invalid}")

    if valid > 0:
        accuracy = (
            valid_df["true_label"] == valid_df["llm_predicted_label"]
        ).mean()

        print(f"\nLLM accuracy on valid predictions only: {accuracy:.4f}")

        strict_accuracy = (
            output_df["true_label"] == output_df["llm_predicted_label"]
        ).mean()

        print(f"LLM strict accuracy including invalid outputs: {strict_accuracy:.4f}")

    print("\nPrediction distribution:")
    print(output_df["llm_predicted_label"].value_counts())


if __name__ == "__main__":
    main()
