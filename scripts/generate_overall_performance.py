#!/usr/bin/env python3
"""
Generate Overall Performance Results for Paper (Results.2)

Runs statistical analysis on the 5 main configurations and generates:
- Performance tables (CSV + LaTeX)
- BA_clip figure with error bars
- Long-format data for reproducibility

Outputs:
- paper_artifacts/tables/Tab_R02_OverallPerformance.csv
- paper_artifacts/tables/Tab_R02_OverallPerformance.tex
- paper_artifacts/data/OverallPerformance_Long.csv
- paper_artifacts/figures/Fig_R01_Overall_BAclip.pdf
- paper_artifacts/figures/Fig_R01_Overall_BAclip.png
- paper_artifacts/data/statistical_run/statistical_analysis.json
- paper_artifacts/data/statistical_run/statistical_report.txt
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from statistical_analysis import (
    load_predictions,
    compute_recalls_with_wilson,
    cluster_bootstrap_ba
)


# Configuration
RESULTS_DIR = Path('/mnt/fast/nobackup/users/gb0048/opro2_clean/results')
OUTPUT_ROOT = Path('/mnt/fast/nobackup/users/gb0048/opro2_clean')

# Main configurations in desired order (4-config design)
CONFIGURATIONS = [
    {
        'key': 'baseline',
        'name': 'Baseline',
        'path': RESULTS_DIR / 'complete_pipeline_seed42/01_baseline/predictions.csv'
    },
    {
        'key': 'base_opro',
        'name': 'Base+OPRO',
        'path': RESULTS_DIR / 'complete_pipeline_seed42/06_eval_base_opro/predictions.csv'
    },
    {
        'key': 'lora',
        'name': 'LoRA',
        'path': RESULTS_DIR / 'complete_pipeline_seed42/03_eval_lora/predictions.csv'
    },
    {
        'key': 'lora_opro',
        'name': 'LoRA+OPRO',
        'path': RESULTS_DIR / 'complete_pipeline_seed42/07_eval_lora_opro/predictions.csv'
    }
]

N_BOOTSTRAP = 10000
RANDOM_SEED = 42


def compute_config_metrics(config_key: str, csv_path: Path) -> dict:
    """
    Compute all metrics for a single configuration.

    Args:
        config_key: Configuration identifier
        csv_path: Path to predictions.csv

    Returns:
        Dictionary with all computed metrics
    """
    print(f"\n{'='*80}")
    print(f"Processing: {config_key}")
    print(f"{'='*80}")
    print(f"Source: {csv_path}")

    # Load predictions
    df = load_predictions(str(csv_path))
    print(f"  Loaded {len(df)} samples ({df['clip_id'].nunique()} unique clips)")

    # Compute BA with cluster bootstrap
    print("  Computing BA_clip with cluster bootstrap...")
    ba_clip, ba_ci_low, ba_ci_high = cluster_bootstrap_ba(
        df,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_SEED
    )

    # Compute per-class recalls with Wilson score intervals
    print("  Computing per-class recalls with Wilson score CIs...")
    recalls = compute_recalls_with_wilson(df)

    metrics = {
        'config_key': config_key,
        'ba_clip': float(ba_clip),
        'ba_clip_ci_low': float(ba_ci_low),
        'ba_clip_ci_high': float(ba_ci_high),
        'recall_speech': float(recalls['recall_speech']),
        'recall_speech_ci_low': float(recalls['recall_speech_ci'][0]),
        'recall_speech_ci_high': float(recalls['recall_speech_ci'][1]),
        'n_speech': int(recalls['n_speech']),
        'recall_nonspeech': float(recalls['recall_nonspeech']),
        'recall_nonspeech_ci_low': float(recalls['recall_nonspeech_ci'][0]),
        'recall_nonspeech_ci_high': float(recalls['recall_nonspeech_ci'][1]),
        'n_nonspeech': int(recalls['n_nonspeech']),
        'n_samples': len(df),
        'n_clips': df['clip_id'].nunique()
    }

    # Print summary
    print(f"\n  Results:")
    print(f"    BA_clip: {ba_clip:.4f} [{ba_ci_low:.4f}, {ba_ci_high:.4f}]")
    print(f"    Recall_Speech: {recalls['recall_speech']:.4f} "
          f"[{recalls['recall_speech_ci'][0]:.4f}, {recalls['recall_speech_ci'][1]:.4f}]")
    print(f"    Recall_NonSpeech: {recalls['recall_nonspeech']:.4f} "
          f"[{recalls['recall_nonspeech_ci'][0]:.4f}, {recalls['recall_nonspeech_ci'][1]:.4f}]")

    return metrics


def export_performance_table(results: list, output_dir: Path):
    """
    Export performance table as CSV and LaTeX.

    Args:
        results: List of metric dictionaries
        output_dir: Output directory (paper_artifacts/tables/)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build wide-format DataFrame
    table_data = []
    for result in results:
        table_data.append({
            'Config': result['config_name'],
            'BA_clip': result['ba_clip'],
            'BA_clip_CI_low': result['ba_clip_ci_low'],
            'BA_clip_CI_high': result['ba_clip_ci_high'],
            'Recall_Speech': result['recall_speech'],
            'Recall_Speech_CI_low': result['recall_speech_ci_low'],
            'Recall_Speech_CI_high': result['recall_speech_ci_high'],
            'Recall_NonSpeech': result['recall_nonspeech'],
            'Recall_NonSpeech_CI_low': result['recall_nonspeech_ci_low'],
            'Recall_NonSpeech_CI_high': result['recall_nonspeech_ci_high']
        })

    df_wide = pd.DataFrame(table_data)

    # Export CSV
    csv_path = output_dir / 'Tab_R02_OverallPerformance.csv'
    df_wide.to_csv(csv_path, index=False)
    print(f"\n✓ Exported CSV: {csv_path}")

    # Export LaTeX (booktabs style)
    latex_path = output_dir / 'Tab_R02_OverallPerformance.tex'

    with open(latex_path, 'w') as f:
        f.write("% Overall Performance Table\n")
        f.write("% Auto-generated - do not edit manually\n\n")
        f.write("\\begin{table*}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Overall Performance Across Configurations}\n")
        f.write("\\label{tab:overall_performance}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Configuration & BA (clip-level) & Recall (SPEECH) & Recall (NONSPEECH) \\\\\n")
        f.write("\\midrule\n")

        for result in results:
            config = result['config_name']
            ba = f"{result['ba_clip']:.2f}"
            ba_ci = f"[{result['ba_clip_ci_low']:.2f}, {result['ba_clip_ci_high']:.2f}]"

            rs = f"{result['recall_speech']:.2f}"
            rs_ci = f"[{result['recall_speech_ci_low']:.2f}, {result['recall_speech_ci_high']:.2f}]"

            rn = f"{result['recall_nonspeech']:.2f}"
            rn_ci = f"[{result['recall_nonspeech_ci_low']:.2f}, {result['recall_nonspeech_ci_high']:.2f}]"

            # Format as: value (95% CI: [low, high])
            ba_str = f"{ba} {ba_ci}"
            rs_str = f"{rs} {rs_ci}"
            rn_str = f"{rn} {rn_ci}"

            f.write(f"{config} & {ba_str} & {rs_str} & {rn_str} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")

    print(f"✓ Exported LaTeX: {latex_path}")


def export_long_format(results: list, output_dir: Path):
    """
    Export long-format data for reproducibility.

    Args:
        results: List of metric dictionaries
        output_dir: Output directory (paper_artifacts/data/)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build long-format DataFrame
    long_data = []

    for result in results:
        config = result['config_name']
        config_key = result['config_key']

        # BA_clip
        long_data.append({
            'config_key': config_key,
            'config_name': config,
            'metric': 'BA_clip',
            'point': result['ba_clip'],
            'ci_low': result['ba_clip_ci_low'],
            'ci_high': result['ba_clip_ci_high']
        })

        # Recall_Speech
        long_data.append({
            'config_key': config_key,
            'config_name': config,
            'metric': 'Recall_Speech',
            'point': result['recall_speech'],
            'ci_low': result['recall_speech_ci_low'],
            'ci_high': result['recall_speech_ci_high']
        })

        # Recall_NonSpeech
        long_data.append({
            'config_key': config_key,
            'config_name': config,
            'metric': 'Recall_NonSpeech',
            'point': result['recall_nonspeech'],
            'ci_low': result['recall_nonspeech_ci_low'],
            'ci_high': result['recall_nonspeech_ci_high']
        })

    df_long = pd.DataFrame(long_data)

    # Export
    csv_path = output_dir / 'OverallPerformance_Long.csv'
    df_long.to_csv(csv_path, index=False)
    print(f"✓ Exported long-format data: {csv_path}")


def generate_ba_clip_figure(results: list, output_dir: Path):
    """
    Generate BA_clip figure with error bars.

    Args:
        results: List of metric dictionaries
        output_dir: Output directory (paper_artifacts/figures/)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    config_names = [r['config_name'] for r in results]
    ba_values = [r['ba_clip'] for r in results]
    ba_ci_low = [r['ba_clip_ci_low'] for r in results]
    ba_ci_high = [r['ba_clip_ci_high'] for r in results]

    # Compute error bar sizes
    yerr_low = [ba_values[i] - ba_ci_low[i] for i in range(len(ba_values))]
    yerr_high = [ba_ci_high[i] - ba_values[i] for i in range(len(ba_values))]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars with error bars
    x_pos = np.arange(len(config_names))
    bars = ax.bar(x_pos, ba_values,
                   color='steelblue',
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=1.2)

    ax.errorbar(x_pos, ba_values,
                yerr=[yerr_low, yerr_high],
                fmt='none',
                ecolor='black',
                capsize=5,
                capthick=1.5,
                linewidth=1.5)

    # Customize plot
    ax.set_ylabel('Balanced Accuracy (BA_clip)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance: Balanced Accuracy Across Configurations',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Chance level')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, ba_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + yerr_high[i] + 0.01,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.legend(loc='lower right')
    plt.tight_layout()

    # Export PDF and PNG
    pdf_path = output_dir / 'Fig_R01_Overall_BAclip.pdf'
    png_path = output_dir / 'Fig_R01_Overall_BAclip.png'

    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"✓ Exported figure (PDF): {pdf_path}")
    print(f"✓ Exported figure (PNG): {png_path}")

    plt.close(fig)


def save_json_report(results: list, output_dir: Path):
    """
    Save statistical analysis results as JSON.

    Args:
        results: List of metric dictionaries
        output_dir: Output directory (paper_artifacts/data/statistical_run/)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build JSON structure
    json_data = {
        'bootstrap_params': {
            'n_bootstrap': N_BOOTSTRAP,
            'random_state': RANDOM_SEED
        },
        'configurations': {}
    }

    for result in results:
        config_key = result['config_key']
        json_data['configurations'][config_key] = {
            'config_name': result['config_name'],
            'n_samples': result['n_samples'],
            'n_clips': result['n_clips'],
            'ba_clip': {
                'point': result['ba_clip'],
                'ci': [result['ba_clip_ci_low'], result['ba_clip_ci_high']]
            },
            'recalls': {
                'recall_speech': result['recall_speech'],
                'recall_speech_ci': [result['recall_speech_ci_low'], result['recall_speech_ci_high']],
                'n_speech': result['n_speech'],
                'recall_nonspeech': result['recall_nonspeech'],
                'recall_nonspeech_ci': [result['recall_nonspeech_ci_low'], result['recall_nonspeech_ci_high']],
                'n_nonspeech': result['n_nonspeech']
            }
        }

    # Save JSON
    json_path = output_dir / 'statistical_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"✓ Exported JSON: {json_path}")


def save_text_report(results: list, output_dir: Path):
    """
    Save human-readable text report.

    Args:
        results: List of metric dictionaries
        output_dir: Output directory (paper_artifacts/data/statistical_run/)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / 'statistical_report.txt'

    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OVERALL PERFORMANCE STATISTICAL REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Bootstrap samples: {N_BOOTSTRAP}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n")
        f.write(f"Configurations: {len(results)}\n\n")

        for result in results:
            f.write("="*80 + "\n")
            f.write(f"{result['config_name']}\n")
            f.write("="*80 + "\n")
            f.write(f"Samples: {result['n_samples']} ({result['n_clips']} clips)\n\n")

            # BA
            f.write(f"Balanced Accuracy (clip-level):\n")
            f.write(f"  BA_clip: {result['ba_clip']:.4f} "
                   f"[{result['ba_clip_ci_low']:.4f}, {result['ba_clip_ci_high']:.4f}]\n\n")

            # Recalls
            f.write(f"Per-class Recalls:\n")
            f.write(f"  Speech: {result['recall_speech']:.4f} "
                   f"[{result['recall_speech_ci_low']:.4f}, {result['recall_speech_ci_high']:.4f}] "
                   f"(n={result['n_speech']})\n")
            f.write(f"  NonSpeech: {result['recall_nonspeech']:.4f} "
                   f"[{result['recall_nonspeech_ci_low']:.4f}, {result['recall_nonspeech_ci_high']:.4f}] "
                   f"(n={result['n_nonspeech']})\n\n")

    print(f"✓ Exported text report: {txt_path}")


def main():
    print("="*80)
    print("OVERALL PERFORMANCE RESULTS GENERATION (Results.2)")
    print("="*80)
    print(f"Configurations: {len(CONFIGURATIONS)}")
    print(f"Bootstrap samples: {N_BOOTSTRAP}")
    print(f"Random seed: {RANDOM_SEED}")

    # Process all configurations
    all_results = []

    for config in CONFIGURATIONS:
        config_key = config['key']
        config_name = config['name']
        csv_path = config['path']

        if not csv_path.exists():
            print(f"\n❌ ERROR: File not found: {csv_path}")
            continue

        # Compute metrics
        metrics = compute_config_metrics(config_key, csv_path)
        metrics['config_name'] = config_name

        all_results.append(metrics)

    if len(all_results) != len(CONFIGURATIONS):
        print(f"\n⚠️  WARNING: Only {len(all_results)}/{len(CONFIGURATIONS)} configs processed")

    # Export all artifacts
    print(f"\n{'='*80}")
    print("Exporting artifacts...")
    print(f"{'='*80}")

    # Tables
    export_performance_table(all_results, OUTPUT_ROOT / 'tables')

    # Long-format data
    export_long_format(all_results, OUTPUT_ROOT / 'data')

    # Figure
    generate_ba_clip_figure(all_results, OUTPUT_ROOT / 'figures')

    # Statistical run outputs
    stat_run_dir = OUTPUT_ROOT / 'data' / 'statistical_run'
    save_json_report(all_results, stat_run_dir)
    save_text_report(all_results, stat_run_dir)

    # Verification log
    print(f"\n{'='*80}")
    print("VERIFICATION LOG")
    print(f"{'='*80}")
    print("\nComparing with statistical_report.txt:")

    for result in all_results:
        print(f"\n{result['config_name']}:")
        print(f"  BA_clip: {result['ba_clip']:.4f} "
              f"[{result['ba_clip_ci_low']:.4f}, {result['ba_clip_ci_high']:.4f}]")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")
    print(f"\nAll artifacts exported to: {OUTPUT_ROOT}")
    print(f"  Tables: {OUTPUT_ROOT / 'tables'}")
    print(f"  Figures: {OUTPUT_ROOT / 'figures'}")
    print(f"  Data: {OUTPUT_ROOT / 'data'}")
    print(f"  Statistical run: {OUTPUT_ROOT / 'data' / 'statistical_run'}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
