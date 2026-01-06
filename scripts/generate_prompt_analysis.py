#!/usr/bin/env python3
"""
Generate prompt analysis artifacts for Results.7
"""
import csv
import json
from pathlib import Path

# Configuration
SEED = 42
OUTPUT_DIR = Path("paper_artifacts")
TABLES_DIR = OUTPUT_DIR / "tables"
DATA_DIR = OUTPUT_DIR / "data"

# Create directories
TABLES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Define prompts for each configuration (4 primary configs only)
prompts = {
    "baseline": "Does this audio contain human speech? Answer SPEECH or NONSPEECH.",
    "base_opro": "Listen briefly; is this clip human speech or noise? Quickly reply: SPEECH or NONSPEECH.",
    "lora": "Does this audio contain human speech? Answer SPEECH or NONSPEECH.",
    "lora_opro": """Decide the dominant content.
Definitions:
- SPEECH = human voice, spoken words, syllables, conversational cues.
- NONSPEECH = music, tones/beeps, environmental noise, silence.
Output exactly: SPEECH or NONSPEECH."""
}

# Analyze each prompt
prompt_data = []
for config, prompt in prompts.items():
    length_chars = len(prompt)

    # Check for explicit definitions
    has_definitions = "Definitions:" in prompt or (
        "SPEECH =" in prompt and "NONSPEECH =" in prompt
    )

    # Create preview (max 80 chars)
    prompt_preview = prompt.replace("\n", " ")[:80]
    if len(prompt.replace("\n", " ")) > 80:
        prompt_preview += "..."

    prompt_data.append({
        "config": config,
        "length_chars": length_chars,
        "has_definitions": has_definitions,
        "prompt_preview": prompt_preview
    })

# Export CSV
csv_path = TABLES_DIR / "Tab_R06_PromptSummary.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["config", "length_chars", "has_definitions", "prompt_preview"])
    writer.writeheader()
    writer.writerows(prompt_data)
print(f"✓ Saved: {csv_path}")

# Export LaTeX table
latex_path = TABLES_DIR / "Tab_R06_PromptSummary.tex"
with open(latex_path, 'w') as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Prompt Summary by Configuration}\n")
    f.write("\\label{tab:prompt_summary}\n")
    f.write("\\begin{tabular}{@{}llcp{6cm}@{}}\n")
    f.write("\\toprule\n")
    f.write("Configuration & Length & Definitions & Preview \\\\\n")
    f.write("\\midrule\n")

    for row in prompt_data:
        config = row["config"].replace("_", "\\_")
        length = row["length_chars"]
        has_def = "Yes" if row["has_definitions"] else "No"
        preview = row["prompt_preview"].replace("_", "\\_").replace("&", "\\&")

        f.write(f"{config} & {length} & {has_def} & {preview} \\\\\n")

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")
print(f"✓ Saved: {latex_path}")

# Export full prompts to appendix text file
appendix_path = DATA_DIR / "Prompts_Appendix.txt"
with open(appendix_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("PROMPT ANALYSIS - FULL PROMPTS BY CONFIGURATION\n")
    f.write("=" * 80 + "\n\n")

    for config, prompt in prompts.items():
        f.write("-" * 80 + "\n")
        f.write(f"Configuration: {config}\n")
        f.write(f"Length: {len(prompt)} characters\n")
        has_def = "Definitions:" in prompt or ("SPEECH =" in prompt and "NONSPEECH =" in prompt)
        f.write(f"Has explicit definitions: {'Yes' if has_def else 'No'}\n")
        f.write("-" * 80 + "\n")
        f.write(prompt)
        f.write("\n\n")
print(f"✓ Saved: {appendix_path}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for row in prompt_data:
    print(f"{row['config']:25s} | Len: {row['length_chars']:3d} | Defs: {str(row['has_definitions']):5s}")
print("=" * 60)
