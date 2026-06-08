# src/12_evaluate_sft_model.py

import json
import re
import time
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


INPUT_PATH = "data/processed/llm_eval_subset.csv"
OUTPUT_PATH = "data/processed/llm_predictions_lora_sft.csv"

MODEL_PATH = "/home/alim/LlamaFactory/models/qwen2.5-3b-llm-final-project-lora-merged"

VALID_LABELS = ["BenignPositive", "FalsePositive", "TruePositive"]


SYSTEM_PROMPT = """You are a cybersecurity SOC alert triage analyst.
Classify the alert into exactly one label:
TruePositive, FalsePositive, or BenignPositive.
Return only valid JSON with this schema:
{"predicted_label":"TruePositive or FalsePositive or BenignPositive"}"""


def build_messages(alert_text: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Security alert:\n{alert_text}\n\nReturn only JSON."
        }
    ]


def extract_json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {"predicted_label": "PARSE_ERROR"}


def normalize_label(label: str) -> str:
    label = str(label).strip()

    mapping = {
        "truepositive": "TruePositive",
        "true positive": "TruePositive",
        "falsepositive": "FalsePositive",
        "false positive": "FalsePositive",
        "benignpositive": "BenignPositive",
        "benign positive": "BenignPositive",
        "benign": "BenignPositive"
    }

    if label in VALID_LABELS:
        return label

    key = label.lower().replace("_", " ").strip()
    compact = key.replace(" ", "")

    if key in mapping:
        return mapping[key]
    if compact in mapping:
        return mapping[compact]

    return "INVALID_LABEL"


def main():
    df = pd.read_csv(INPUT_PATH)

    print(f"Loading model from: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        messages = build_messages(row["alert_text"])

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.time()

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id
            )

        latency = time.time() - start

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        parsed = extract_json(raw_response)
        pred_raw = parsed.get("predicted_label", "INVALID_LABEL")
        pred = normalize_label(pred_raw)

        results.append({
            "Id": row["Id"],
            "IncidentId": row["IncidentId"],
            "AlertId": row["AlertId"],
            "Timestamp": row["Timestamp"],
            "alert_text": row["alert_text"],
            "true_label": row["label"],
            "llm_predicted_label_raw": pred_raw,
            "llm_predicted_label": pred,
            "raw_response": raw_response,
            "latency_seconds": latency
        })

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_PATH, index=False)

    y_true = out["true_label"]
    y_pred = out["llm_predicted_label"]

    print(f"\nSaved predictions to: {OUTPUT_PATH}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=VALID_LABELS))

    print("\nConfusion matrix:")
    print("Labels:", VALID_LABELS)
    print(confusion_matrix(y_true, y_pred, labels=VALID_LABELS))


if __name__ == "__main__":
    main()
