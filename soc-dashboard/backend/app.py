# backend/app.py

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "tfidf_logreg_baseline.joblib"
SAMPLE_PATH = BASE_DIR / "data" / "processed" / "llm_eval_subset.csv"
FRONTEND_DIR = BASE_DIR / "frontend"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"

HOST = "0.0.0.0"
PORT = 8000


# ============================================================
# Request / Response Schemas
# ============================================================

class AlertRequest(BaseModel):
    alert_text: str


class HybridTriageResponse(BaseModel):
    alert_text: str
    ml_predicted_label: str
    analyst_summary: str
    risk_explanation: str
    mitre_interpretation: List[str]
    recommended_actions: List[str]
    containment_priority: str
    raw_llm_response: Optional[str] = None
    latency_seconds: float


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="LLM-SOC Hybrid Alert Triage API",
    description="FastAPI backend and frontend for ML + Qwen SOC alert triage",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


classifier = None
sample_df = None


# ============================================================
# Startup Resource Loading
# ============================================================

@app.on_event("startup")
def load_resources() -> None:
    global classifier, sample_df

    print("=" * 80)
    print("Starting LLM-SOC Hybrid Alert Triage Backend")
    print("=" * 80)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Sample data not found: {SAMPLE_PATH}")

    if not FRONTEND_DIR.exists():
        raise FileNotFoundError(f"Frontend folder not found: {FRONTEND_DIR}")

    classifier = joblib.load(MODEL_PATH)
    sample_df = pd.read_csv(SAMPLE_PATH)

    print(f"Loaded classifier: {MODEL_PATH}")
    print(f"Loaded sample data: {SAMPLE_PATH}")
    print(f"Loaded frontend: {FRONTEND_DIR}")
    print(f"Sample rows: {len(sample_df)}")
    print(f"Ollama model: {OLLAMA_MODEL}")
    print("=" * 80)


# ============================================================
# Helper Functions
# ============================================================

def extract_json(text: str) -> Dict[str, Any]:
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
        "analyst_summary": "Could not parse LLM output as JSON.",
        "risk_explanation": "",
        "mitre_interpretation": [],
        "recommended_actions": [],
        "containment_priority": "Unknown",
    }


def safe_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]

    if isinstance(value, str) and value.strip():
        return [value]

    return []


def build_hybrid_prompt(alert_text: str, ml_prediction: str) -> str:
    return f"""
You are a cybersecurity SOC analyst.

A machine learning classifier has already predicted this alert as:

{ml_prediction}

Your job is NOT to change the label.
Your job is to generate an analyst-facing explanation and response guidance.

Return ONLY valid JSON.
Do not include markdown.
Do not include text outside JSON.

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


def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 4096,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()
    return response.json()["response"]


# ============================================================
# API Routes
# ============================================================

@app.get("/api")
def api_root():
    return {
        "message": "LLM-SOC Hybrid Alert Triage API is running",
        "backend": "FastAPI",
        "frontend": "/",
        "docs": "/docs",
        "llm_model": OLLAMA_MODEL,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "classifier_loaded": classifier is not None,
        "sample_rows": 0 if sample_df is None else len(sample_df),
        "ollama_model": OLLAMA_MODEL,
    }


@app.get("/samples")
def get_samples(limit: int = 20):
    if sample_df is None:
        raise HTTPException(status_code=500, detail="Sample data not loaded")

    limit = max(1, min(limit, 100))

    cols = ["Id", "IncidentId", "AlertId", "Timestamp", "alert_text", "label"]
    available_cols = [col for col in cols if col in sample_df.columns]

    records = sample_df[available_cols].head(limit).to_dict(orient="records")

    return {
        "count": len(records),
        "samples": records,
    }


@app.get("/samples/{index}")
def get_sample_by_index(index: int):
    if sample_df is None:
        raise HTTPException(status_code=500, detail="Sample data not loaded")

    if index < 0 or index >= len(sample_df):
        raise HTTPException(status_code=404, detail="Sample index out of range")

    return sample_df.iloc[index].to_dict()


@app.post("/predict")
def predict_only(request: AlertRequest):
    if classifier is None:
        raise HTTPException(status_code=500, detail="Classifier not loaded")

    alert_text = request.alert_text.strip()

    if not alert_text:
        raise HTTPException(status_code=400, detail="alert_text cannot be empty")

    try:
        prediction = classifier.predict([alert_text])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML prediction failed: {e}")

    return {
        "alert_text": alert_text,
        "ml_predicted_label": prediction,
    }


@app.post("/triage", response_model=HybridTriageResponse)
def triage_alert(request: AlertRequest):
    if classifier is None:
        raise HTTPException(status_code=500, detail="Classifier not loaded")

    alert_text = request.alert_text.strip()

    if not alert_text:
        raise HTTPException(status_code=400, detail="alert_text cannot be empty")

    try:
        ml_prediction = classifier.predict([alert_text])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML prediction failed: {e}")

    prompt = build_hybrid_prompt(alert_text, ml_prediction)

    start = time.time()

    try:
        raw_llm_response = call_ollama(prompt)
        parsed = extract_json(raw_llm_response)
    except Exception as e:
        raw_llm_response = str(e)
        parsed = {
            "analyst_summary": f"Ollama request failed: {e}",
            "risk_explanation": "",
            "mitre_interpretation": [],
            "recommended_actions": [],
            "containment_priority": "Unknown",
        }

    latency = time.time() - start

    return HybridTriageResponse(
        alert_text=alert_text,
        ml_predicted_label=ml_prediction,
        analyst_summary=str(parsed.get("analyst_summary", "")),
        risk_explanation=str(parsed.get("risk_explanation", "")),
        mitre_interpretation=safe_list(parsed.get("mitre_interpretation", [])),
        recommended_actions=safe_list(parsed.get("recommended_actions", [])),
        containment_priority=str(parsed.get("containment_priority", "Unknown")),
        raw_llm_response=raw_llm_response,
        latency_seconds=latency,
    )


# ============================================================
# Serve Frontend
# ============================================================
# Important:
# Mount this AFTER all API routes.
# This makes http://localhost:8000 load frontend/index.html.

app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend",
)


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        reload=False,
    )
