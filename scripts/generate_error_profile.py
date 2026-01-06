#!/usr/bin/env python3
"""
Generate error profile and recall trade-off analysis (Results.6)
Creates:
- Tab_R05_ErrorCounts.csv + .tex
- Fig_R08_Recall_Tradeoff.pdf + .png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
SEED = 42
RESULTS_BASE = Path("results")
OUTPUT_DIR = Path("paper_artifacts")
OUTPUT_DIR_TABLES = OUTPUT_DIR / "tables"
OUTPUT_DIR_FIGURES = OUTPUT_DIR / "figures"

# Create output directories
OUTPUT_DIR_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_FIGURES.mkdir(parents=True, exist_ok=True)

# Configuration paths in order
CONFIGS = {
    "baseline": RESULTS_BASE / "complete_pipeline_seed42" / "01_baseline" / "predictions.csv",
    "base_opro_classic": RESULTS_BASE / "complete_pipeline_seed42" / "06_eval_base_opro" / "predictions.csv",
    "lora_hand": RESULTS_BASE / "complete_pipeline_seed42" / "03_eval_lora" / "predictions.csv",
    "lora_opro_classic": RESULTS_BASE / "complete_pipeline_seed42" / "07_eval_lora_opro" / "predictions.csv",
    "lora_opro_open": RESULTS_BASE / "complete_pipeline_seed42_opro_open" / "07_eval_lora_opro" / "predictions.csv",
}

# Pretty names for figures
CONFIG_LABELS = {
    "baseline": "Baseline",
    "base_opro_classic": "Base+OPRO",
    "lora_hand": "LoRA+Hand",
    "lora_opro_classic": "LoRA+OPRO",
    "lora_opro_open": "LoRA+OPRO-Open",
}


def compute_error_metrics(df):
    """
    Compute error metrics treating SPEECH as positive class.

    Returns dict with: TP, TN, FP, FN, FPR, FNR, Recall_Speech, Recall_NonSpeech
    """
    # Confusion matrix components (SPEECH = positive)
    tp = ((df['ground_truth'] == 'SPEECH') & (df['prediction'] == 'SPEECH')).sum()
    tn = ((df['ground_truth'] == 'NONSPEECH') & (df['prediction'] == 'NONSPEECH')).sum()
    fp = ((df['ground_truth'] == 'NONSPEECH') & (df['prediction'] == 'SPEECH')).sum()
    fn = ((df['ground_truth'] == 'SPEECH') & (df['prediction'] == 'NONSPEECH')).sum()

    # Rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Recalls
    recall_speech = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR, Sensitivity
    recall_nonspeech = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR, Specificity

    return {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'FPR': fpr,
        'FNR': fnr,
        'Recall_Speech': recall_speech,
        'Recall_NonSpeech': recall_nonspeech,
    }


def main():
    print("Generating error profile and recall trade-off analysis...")

    # Collect metrics for all configs
    results = []

    for config_name, pred_path in CONFIGS.items():
        print(f"\nProcessing {config_name}...")

        if not pred_path.exists():
            print(f"  WARNING: {pred_path} not found, skipping")
            continue

        # Load predictions
        df = pd.read_csv(pred_path)
        print(f"  Loaded {len(df)} predictions")

        # Compute metrics
        metrics = compute_error_metrics(df)
        metrics['config'] = config_name
        results.append(metrics)

        print(f"  TP={metrics['TP']}, TN={metrics['TN']}, "
              f"FP={metrics['FP']}, FN={metrics['FN']}")
        print(f"  FPR={metrics['FPR']:.4f}, FNR={metrics['FNR']:.4f}")
        print(f"  Recall_Speech={metrics['Recall_Speech']:.4f}, "
              f"Recall_NonSpeech={metrics['Recall_NonSpeech']:.4f}")

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Reorder columns
    results_df = results_df[['config', 'TP', 'TN', 'FP', 'FN',
                              'FPR', 'FNR', 'Recall_Speech', 'Recall_NonSpeech']]

    # Save CSV
    csv_path = OUTPUT_DIR_TABLES / "Tab_R05_ErrorCounts.csv"
    results_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n✓ Saved CSV: {csv_path}")

    # Generate LaTeX table
    latex_path = OUTPUT_DIR_TABLES / "Tab_R05_ErrorCounts.tex"

    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Global Error Profile by Configuration}\n")
        f.write("\\label{tab:error-counts}\n")
        f.write("\\begin{tabular}{lrrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Config & TP & TN & FP & FN & FPR & FNR & Recall$_{\\text{Speech}}$ & Recall$_{\\text{NonSpeech}}$ \\\\\n")
        f.write("\\midrule\n")

        for _, row in results_df.iterrows():
            config_label = CONFIG_LABELS.get(row['config'], row['config'])
            f.write(f"{config_label} & "
                   f"{int(row['TP'])} & {int(row['TN'])} & "
                   f"{int(row['FP'])} & {int(row['FN'])} & "
                   f"{row['FPR']:.3f} & {row['FNR']:.3f} & "
                   f"{row['Recall_Speech']:.3f} & {row['Recall_NonSpeech']:.3f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"✓ Saved LaTeX: {latex_path}")

    # Generate recall trade-off figure
    print("\nGenerating recall trade-off plot...")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot points
    for _, row in results_df.iterrows():
        config_label = CONFIG_LABELS.get(row['config'], row['config'])
        ax.scatter(row['Recall_NonSpeech'], row['Recall_Speech'],
                  s=150, alpha=0.7, label=config_label)

        # Add text label
        ax.text(row['Recall_NonSpeech'] + 0.005, row['Recall_Speech'] + 0.005,
               config_label, fontsize=9, ha='left')

    # Formatting
    ax.set_xlabel('Recall (Non-Speech)', fontsize=12)
    ax.set_ylabel('Recall (Speech)', fontsize=12)
    ax.set_title('Recall Trade-off by Configuration', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set limits with some padding
    x_min = results_df['Recall_NonSpeech'].min() - 0.02
    x_max = results_df['Recall_NonSpeech'].max() + 0.08
    y_min = results_df['Recall_Speech'].min() - 0.02
    y_max = results_df['Recall_Speech'].max() + 0.02

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Add diagonal reference line (equal recall)
    lim_min = max(x_min, y_min)
    lim_max = min(x_max, y_max)
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
           'k--', alpha=0.3, linewidth=1, label='Equal Recall')

    plt.tight_layout()

    # Save PDF
    pdf_path = OUTPUT_DIR_FIGURES / "Fig_R08_Recall_Tradeoff.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PDF: {pdf_path}")

    # Save PNG
    png_path = OUTPUT_DIR_FIGURES / "Fig_R08_Recall_Tradeoff.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PNG: {png_path}")

    plt.close()

    print("\n" + "="*60)
    print("Error profile analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
