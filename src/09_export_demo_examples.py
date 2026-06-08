# src/09_export_demo_examples.py

import pandas as pd

INPUT_PATH = "data/processed/hybrid_ml_llm_outputs.csv"
OUTPUT_PATH = "reports/demo_hybrid_examples.md"

df = pd.read_csv(INPUT_PATH)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("# Hybrid ML + Qwen Demo Examples\n\n")

    for i, row in df.head(5).iterrows():
        f.write(f"## Example {i+1}\n\n")
        f.write(f"**True Label:** {row['true_label']}\n\n")
        f.write(f"**ML Prediction:** {row['ml_predicted_label']}\n\n")
        f.write(f"**Correct:** {row['is_correct']}\n\n")
        f.write("**Alert Text:**\n\n")
        f.write(f"`{row['alert_text']}`\n\n")
        f.write("**Analyst Summary:**\n\n")
        f.write(f"{row['analyst_summary']}\n\n")
        f.write("**Risk Explanation:**\n\n")
        f.write(f"{row['risk_explanation']}\n\n")
        f.write("**Recommended Actions:**\n\n")
        f.write(f"{row['recommended_actions']}\n\n")
        f.write("---\n\n")

print(f"Saved demo examples to {OUTPUT_PATH}")
