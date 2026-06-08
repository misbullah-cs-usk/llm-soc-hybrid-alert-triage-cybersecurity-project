# LLM-SOC: Hybrid Alert Triage and Response Guidance using Public Security Incident Data

Course Name: Applying LLMs in Cybersecurity Systems
Group Number: 第12組
Group Name: CyberDefense
Team Members:
- Alim Misbullah
- Laina Farsiah
- Moh. Jabir Mubarok 

This repository contains the experimental pipeline for **SOC alert triage using public security incident data, classical machine learning, Qwen LLM prompting, and LoRA-SFT fine-tuning**.

The project evaluates whether Large Language Models can classify SOC alerts directly and whether fine-tuning improves their performance. It also compares LLM-based methods with traditional rule-based and classical machine-learning baselines.

> Note: The SOC Dashboard has a separate README because it uses a different FastAPI + JavaScript application layer.

---

## Project Overview

Security Operations Centers handle large numbers of alerts every day. Analysts must determine whether an alert is a real threat, a false alarm, or benign activity. This project focuses on classifying alerts into three incident-grade labels:

- `TruePositive`
- `FalsePositive`
- `BenignPositive`

The project compares the following methods:

| Method | Purpose |
|---|---|
| Rule-Based Baseline | Traditional SOC-style heuristic logic |
| TF-IDF + Logistic Regression | Classical ML benchmark |
| Qwen Zero-Shot | Direct LLM classification without examples |
| Qwen Few-Shot | Prompt-engineered LLM classification with examples |
| Qwen LoRA-SFT | Task-specific fine-tuned LLM |
| Hybrid ML + Qwen | ML prediction with LLM explanation generation |

---

## Dataset

The project uses the Microsoft security incident prediction dataset.

Main files:

```text
data/raw/GUIDE_Train.csv
data/raw/GUIDE_Test.csv
````

The target column is:

```text
IncidentGrade
```

The selected labels are:

| Label            | Meaning                                                       |
| ---------------- | ------------------------------------------------------------- |
| `TruePositive`   | Alert likely represents real malicious or suspicious activity |
| `FalsePositive`  | Alert was likely triggered incorrectly                        |
| `BenignPositive` | Alert is valid but likely benign, expected, or low-risk       |

---

## Repository Structure

```text
llm-soc-hybrid-alert-triage-cybersecurity-project/
├── data/
│   ├── raw/
│   │   ├── GUIDE_Train.csv
│   │   └── GUIDE_Test.csv
│   ├── processed/
│   │   ├── train_processed.csv
│   │   ├── test_processed.csv
│   │   ├── llm_eval_subset.csv
│   │   ├── baseline_predictions.csv
│   │   ├── llm_predictions_ollama.csv
│   │   ├── llm_predictions_fewshot_ollama.csv
│   │   └── llm_predictions_lora_sft.csv
│   └── sft/
│       ├── soc_triage_sft_train.json
│       ├── soc_triage_sft_val.json
│       └── soc_triage_sft_test.json
├── models/
│   └── tfidf_logreg_baseline.joblib
├── reports/
│   └── all_model_comparison.csv
├── src/
│   ├── 01_explore_dataset.py
│   ├── 02_preprocess.py
│   ├── 03_train_rule_based_baseline.py
│   ├── 03_train_baseline.py
│   ├── 04_create_llm_eval_subset.py
│   ├── 04_evaluate_baselines_on_llm_subset.py
│   ├── 05_llm_triage_zeroshot_ollama.py
│   ├── 05_llm_triage_fewshot_ollama.py
│   ├── 06_evaluate_llm_predictions.py
│   ├── 07_compare_all_models.py
│   ├── 08_hybrid_ml_llm_poc.py
│   ├── 09_export_demo_examples.py
│   ├── 10_prepare_sft_dataset.py
│   └── 11_evaluate_sft_model.py
├── requirements.txt
└── README.md
```

---

## Environment Setup

Create and activate a Python environment:

```bash
python3 -m venv env-llm-soc
source env-llm-soc/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
pandas
numpy
scikit-learn
matplotlib
tqdm
requests
joblib
python-dotenv
torch
transformers
accelerate
```

For LLM experiments, install and run Ollama separately.

Check available Ollama models:

```bash
ollama list
```

Models used in this project:

```text
qwen2.5:3b-instruct
qwen2.5:7b-instruct
qwen3:4b
```

Start Ollama:

```bash
ollama serve
```

---

## Experiment Pipeline

### Step 1: Explore Dataset

Inspect raw CSV files, column names, and sample rows.

```bash
python src/01_explore_dataset.py
```

Purpose:

* Check available dataset files.
* Inspect column names.
* Preview raw records.
* Identify target and useful features.

---

### Step 2: Preprocess Dataset

Convert raw alert records into a structured text format called `alert_text`.

```bash
python src/02_preprocess.py
```

Output:

```text
data/processed/train_processed.csv
data/processed/test_processed.csv
```

Example `alert_text`:

```text
Category: CredentialAccess | MitreTechniques: T1110;T1078 | EntityType: User | EvidenceRole: Impacted | DetectorId: 468 | AlertTitle: 24 | SuspicionLevel: unknown | LastVerdict: unknown | CountryCode: 242 | State: 1445 | City: 10630 | OSFamily: 5 | OSVersion: 66
```

---

### Step 3: Run Rule-Based Baseline

Run the traditional SOC-style heuristic baseline.

```bash
python src/03a_rule_based_baseline.py
```

Output:

```text
data/processed/rule_based_predictions.csv
```

Full test-set result:

| Model               | Accuracy | Macro F1 | Weighted F1 |
| ------------------- | -------: | -------: | ----------: |
| Rule-Based Baseline |   0.3444 |     0.28 |        0.31 |

---

### Step 4: Train TF-IDF + Logistic Regression Baseline

Train the classical machine-learning baseline.

```bash
python src/03_train_baseline.py
```

Output:

```text
models/tfidf_logreg_baseline.joblib
data/processed/baseline_predictions.csv
```

Full test-set result:

| Model                        | Accuracy | Macro F1 | Weighted F1 |
| ---------------------------- | -------: | -------: | ----------: |
| TF-IDF + Logistic Regression |   0.7757 |     0.76 |        0.77 |

---

### Step 5: Create Balanced LLM Evaluation Subset

Create a fixed 300-row balanced subset for LLM experiments.

```bash
python src/04_create_llm_eval_subset.py
```

Output:

```text
data/processed/llm_eval_subset.csv
```

Subset distribution:

| Class          | Rows |
| -------------- | ---: |
| BenignPositive |  100 |
| FalsePositive  |  100 |
| TruePositive   |  100 |
| Total          |  300 |

This same subset is used for:

* Rule-based same-subset evaluation
* Logistic Regression same-subset evaluation
* Qwen zero-shot
* Qwen few-shot
* Qwen LoRA-SFT
* Hybrid ML + Qwen comparison

---

### Step 6: Run Qwen Zero-Shot LLM Classification

Run Qwen through Ollama using zero-shot prompting.

```bash
python src/05_llm_triage_ollama.py
```

Output:

```text
data/processed/llm_predictions_ollama.csv
```

Zero-shot result:

| Metric             |        Value |
| ------------------ | -----------: |
| Evaluation rows    |          300 |
| Valid JSON outputs |          300 |
| Parse errors       |            0 |
| Accuracy           |       0.3600 |
| Macro F1           |       0.3238 |
| Average latency    | 2.11 seconds |

---

### Step 7: Run Qwen Few-Shot LLM Classification

Run Qwen using few-shot examples in the prompt.

```bash
python src/05b_llm_triage_fewshot_ollama.py
```

Output:

```text
data/processed/llm_predictions_fewshot_ollama.csv
```

Few-shot result:

| Model          | Accuracy | Macro F1 |
| -------------- | -------: | -------: |
| Qwen Zero-Shot |   0.3600 |   0.3238 |
| Qwen Few-Shot  |   0.3867 |   0.3659 |

Few-shot prompting improved Qwen slightly, but performance remained below the classical ML baseline.

---

### Step 8: Prepare LoRA-SFT Dataset

Convert processed alert records into Alpaca-style instruction data.

```bash
python src/11_prepare_sft_dataset.py
```

Output:

```text
data/sft/soc_triage_sft_train.json
data/sft/soc_triage_sft_val.json
data/sft/soc_triage_sft_test.json
```

Example SFT format:

```json
{
  "instruction": "You are a cybersecurity SOC alert triage analyst. Classify the alert into exactly one label: TruePositive, FalsePositive, or BenignPositive. Return only valid JSON.",
  "input": "Security alert: Category: CredentialAccess | MitreTechniques: T1110;T1078 | EntityType: User | EvidenceRole: Impacted | DetectorId: 468 | AlertTitle: 24 | SuspicionLevel: unknown | LastVerdict: unknown | CountryCode: 242 | State: 1445 | City: 10630 | OSFamily: 5 | OSVersion: 66",
  "output": "{\"predicted_label\":\"BenignPositive\"}"
}
```

The LLM evaluation subset is excluded from SFT training to reduce data leakage risk.

---

## LoRA-SFT with LLaMA-Factory

The LoRA-SFT experiment uses LLaMA-Factory and Qwen2.5 Instruct.

Example model:

```text
Qwen/Qwen2.5-3B-Instruct
```

Example LLaMA-Factory configuration:

```yaml
model_name_or_path: Qwen/Qwen2.5-3B-Instruct
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05

dataset: soc_triage_sft_train
eval_dataset: soc_triage_sft_val
template: qwen
cutoff_len: 768
max_samples: 12000
overwrite_cache: true

output_dir: saves/qwen2.5-3b/lora/soc_triage
logging_steps: 20
save_steps: 500
plot_loss: true
overwrite_output_dir: true

per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true

val_size: 0.0
per_device_eval_batch_size: 2
eval_strategy: steps
eval_steps: 250
```

Important note:

```yaml
template: qwen
```

Use `qwen` for Qwen2.5 Instruct. Do not use `qwen_nothink` if the installed LLaMA-Factory version does not support it.

---

### Step 9: Evaluate LoRA-SFT Model

After training and merging the LoRA model, evaluate it on the same 300-row subset.

```bash
python src/12_evaluate_sft_model.py
```

Output:

```text
data/processed/llm_predictions_lora_sft.csv
```

LoRA-SFT result:

| Metric          |  Value |
| --------------- | -----: |
| Accuracy        | 0.4833 |
| Macro Precision |   0.61 |
| Macro Recall    |   0.48 |
| Macro F1        |   0.42 |
| Weighted F1     |   0.42 |

Classification report:

| Class          | Precision | Recall | F1-score | Support |
| -------------- | --------: | -----: | -------: | ------: |
| BenignPositive |      0.53 |   0.76 |     0.62 |     100 |
| FalsePositive  |      0.41 |   0.61 |     0.49 |     100 |
| TruePositive   |      0.89 |   0.08 |     0.15 |     100 |

Confusion matrix:

| True Label \ Predicted Label | BenignPositive | FalsePositive | TruePositive |
| ---------------------------- | -------------: | ------------: | -----------: |
| BenignPositive               |             76 |            24 |            0 |
| FalsePositive                |             38 |            61 |            1 |
| TruePositive                 |             30 |            62 |            8 |

---

## Final Model Comparison

Run:

```bash
python src/07_compare_all_models.py
```

Output:

```text
reports/all_model_comparison.csv
```

Final same-subset comparison:

| Model                        | Role                                    | Accuracy | Macro F1 |
| ---------------------------- | --------------------------------------- | -------: | -------: |
| Rule-Based Baseline          | Traditional heuristic SOC logic         |   0.3300 |   0.2899 |
| Qwen Zero-Shot               | Direct LLM classification               |   0.3600 |   0.3238 |
| Qwen Few-Shot                | Prompt-engineered LLM classification    |   0.3867 |   0.3659 |
| Qwen LoRA-SFT                | Task-specific fine-tuned LLM classifier |   0.4833 |   0.4200 |
| TF-IDF + Logistic Regression | Classical ML classifier                 |   0.7967 |   0.7952 |
| Hybrid ML + Qwen             | ML prediction with LLM explanation      |   0.7967 |   0.7952 |

---

## Key Findings

The rule-based baseline performed poorly, showing that static heuristic rules are not enough for this alert triage task.

TF-IDF + Logistic Regression achieved the strongest classification performance because it can learn statistical patterns from encoded fields such as `DetectorId`, `AlertTitle`, and location identifiers.

Qwen zero-shot and few-shot prompting produced valid JSON outputs, but their classification accuracy remained limited because many dataset values are anonymized and not semantically meaningful.

LoRA-SFT improved the LLM result from 0.3600 zero-shot accuracy to 0.4833 accuracy, showing that task-specific fine-tuning helps the model learn the dataset better.

The final hybrid design uses Logistic Regression for prediction and Qwen for explanation and response guidance.

---

## Reproducibility

Recommended execution order:

```bash
python src/01_explore_dataset.py
python src/02_preprocess.py
python src/03a_rule_based_baseline.py
python src/03_train_baseline.py
python src/04_create_llm_eval_subset.py
python src/05_llm_triage_ollama.py
python src/05b_llm_triage_fewshot_ollama.py
python src/11_prepare_sft_dataset.py
python src/12_evaluate_sft_model.py
python src/07_compare_all_models.py
python src/08_hybrid_ml_llm_poc.py
python src/09_export_demo_examples.py
```

---

## Main Outputs

| File                                 | Description                            |
| ------------------------------------ | -------------------------------------- |
| `train_processed.csv`                | Processed training data                |
| `test_processed.csv`                 | Processed test data                    |
| `llm_eval_subset.csv`                | Balanced 300-row LLM evaluation subset |
| `tfidf_logreg_baseline.joblib`       | Saved Logistic Regression pipeline     |
| `llm_predictions_ollama.csv`         | Qwen zero-shot predictions             |
| `llm_predictions_fewshot_ollama.csv` | Qwen few-shot predictions              |
| `llm_predictions_lora_sft.csv`       | LoRA-SFT model predictions             |
| `all_model_comparison.csv`           | Final model comparison table           |
| `hybrid_ml_llm_outputs.csv`          | Hybrid ML + LLM outputs                |
| `demo_hybrid_examples.md`            | Example hybrid outputs for report/demo |

---

## Notes

The LLM evaluation uses a smaller balanced subset because local LLM inference is slower than classical machine-learning inference.

The LoRA-SFT model improves over prompt-only Qwen but still does not outperform Logistic Regression.

The hybrid approach is the most practical result of this experiment because it combines accurate classical ML prediction with LLM-generated analyst explanation.

---

## References

FastAPI. (2026). *FastAPI documentation*. [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. *arXiv*. [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

LLaMA-Factory. (2026). *LLaMA-Factory: Unified efficient fine-tuning of large language models*. GitHub. [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

Microsoft. (2025). *Microsoft Security Incident Prediction*. Kaggle. [https://www.kaggle.com/datasets/microsoft/microsoft-security-incident-prediction](https://www.kaggle.com/datasets/microsoft/microsoft-security-incident-prediction)

MITRE. (2026). *MITRE ATT&CK®*. [https://attack.mitre.org/](https://attack.mitre.org/)

Ollama. (2026). *Ollama API documentation*. [https://docs.ollama.com/api](https://docs.ollama.com/api)

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Qwen Team. (2025). *Qwen2.5 technical report*. *arXiv*. [https://arxiv.org/abs/2412.15115](https://arxiv.org/abs/2412.15115)

```
```

