#!/usr/bin/env python3
"""
Generate publication-quality figures for statistical analysis report.

Generates:
  - Figure 1: Bar plot of BA_clip with 95% CI error bars
  - Figure 2: Forest plot of pairwise comparisons (ΔBA with CI)
  - Figure 3: Psychometric thresholds (DT90) comparison
  - Figure 4: Confusion matrix heatmap for best model
  - Figure 5: ROC-style plot (Recall_Speech vs Recall_NonSpeech)
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Publication settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# Paths
RESULTS_DIR = Path('/mnt/fast/nobackup/users/gb0048/opro2_clean/results')
STATS_FILE = RESULTS_DIR / 'statistical_analysis/statistical_analysis.json'
PSYCHO_DIR = RESULTS_DIR / 'psychometric_analysis'
OUTPUT_DIR = RESULTS_DIR / 'figures'

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Load data
with open(STATS_FILE, 'r') as f:
    stats = json.load(f)

# Model display names (for publication)
MODEL_NAMES = {
    'baseline': 'Baseline',
    'base_opro': 'Base+OPRO',
    'lora': 'LoRA+Hand',
    'lora_opro_classic': 'LoRA+OPRO (Classic)',
    'lora_opro_open': 'LoRA+OPRO (Open)',
    'lora_opro': 'LoRA+OPRO'
}

# Colors for models
COLORS = {
    'baseline': '#e74c3c',      # Red
    'base_opro': '#f39c12',     # Orange
    'lora': '#3498db',          # Blue
    'lora_opro_classic': '#2ecc71',  # Green
    'lora_opro_open': '#27ae60',     # Dark green
    'lora_opro': '#2ecc71'
}


def figure1_ba_barplot():
    """Figure 1: Balanced Accuracy comparison with 95% CI."""

    configs = stats['config_metrics']

    # Order models by performance
    models = ['baseline', 'base_opro', 'lora', 'lora_opro_classic']
    ba_values = [configs[m]['ba_clip'] for m in models]
    ci_lower = [configs[m]['ba_clip_ci'][0] for m in models]
    ci_upper = [configs[m]['ba_clip_ci'][1] for m in models]

    errors = [
        [ba_values[i] - ci_lower[i] for i in range(len(models))],
        [ci_upper[i] - ba_values[i] for i in range(len(models))]
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    bars = ax.bar(x, ba_values,
                   color=[COLORS[m] for m in models],
                   edgecolor='black', linewidth=1.2,
                   alpha=0.8)

    ax.errorbar(x, ba_values, yerr=errors, fmt='none',
                ecolor='black', capsize=5, capthick=1.5, linewidth=1.5)

    ax.set_ylabel('Balanced Accuracy (BA_clip)', fontweight='bold')
    ax.set_xlabel('Model Configuration', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=15, ha='right')
    ax.set_ylim([0.5, 1.0])
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, label='Chance level')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, ba_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax.legend(loc='lower right')
    ax.set_title('Balanced Accuracy by Model Configuration', fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure1_ba_comparison.png')
    plt.savefig(OUTPUT_DIR / 'figure1_ba_comparison.pdf')
    print(f"✓ Figure 1 saved: {OUTPUT_DIR}/figure1_ba_comparison.png")
    plt.close()


def figure2_forest_plot():
    """Figure 2: Forest plot of pairwise comparisons (ΔBA with CI)."""

    comparisons = stats['comparisons']

    comp_labels = [
        'Baseline vs\nBase+OPRO',
        'Baseline vs\nLoRA+Hand',
        'LoRA+Hand vs\nLoRA+OPRO',
        'LoRA+OPRO Classic\nvs Open'
    ]

    deltas = [c['delta_ba'] for c in comparisons]
    ci_lower = [c['delta_ba_ci'][0] for c in comparisons]
    ci_upper = [c['delta_ba_ci'][1] for c in comparisons]
    p_values = [c['p_value_adjusted'] for c in comparisons]

    fig, ax = plt.subplots(figsize=(8, 6))

    y = np.arange(len(comparisons))

    # Colors based on significance
    colors = ['green' if p < 0.05 else 'gray' for p in p_values]

    # Plot points and error bars
    ax.errorbar(deltas, y,
                xerr=[[deltas[i] - ci_lower[i] for i in range(len(deltas))],
                      [ci_upper[i] - deltas[i] for i in range(len(deltas))]],
                fmt='o', markersize=8, capsize=6, capthick=2,
                color='black', ecolor=colors, linewidth=2)

    # Add points with colors
    for i, (delta, color) in enumerate(zip(deltas, colors)):
        ax.plot(delta, i, 'o', markersize=10, color=color,
                markeredgecolor='black', markeredgewidth=1.5)

    # Zero line (no difference)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='No difference')

    ax.set_yticks(y)
    ax.set_yticklabels(comp_labels)
    ax.set_xlabel('Difference in BA (ΔBA)', fontweight='bold')
    ax.set_title('Pairwise Model Comparisons (with 95% CI)', fontweight='bold', pad=15)

    # Legend
    sig_patch = mpatches.Patch(color='green', label='Significant (p < 0.05)')
    ns_patch = mpatches.Patch(color='gray', label='Not significant')
    ax.legend(handles=[sig_patch, ns_patch], loc='lower right')

    # Add p-value annotations
    for i, p in enumerate(p_values):
        p_text = f"p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(ax.get_xlim()[1] * 0.95, i, p_text,
                ha='right', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure2_forest_plot.png')
    plt.savefig(OUTPUT_DIR / 'figure2_forest_plot.pdf')
    print(f"✓ Figure 2 saved: {OUTPUT_DIR}/figure2_forest_plot.png")
    plt.close()


def figure3_psychometric_thresholds():
    """Figure 3: Psychometric thresholds (DT90) comparison."""

    # Load psychometric data
    models = ['baseline', 'base_opro', 'lora_hand', 'lora_opro_classic']
    model_files = {
        'baseline': 'baseline_psychometric.json',
        'base_opro': 'base_opro_classic_psychometric.json',
        'lora_hand': 'lora_hand_psychometric.json',
        'lora_opro_classic': 'lora_opro_classic_psychometric.json'
    }

    dt90_values = []
    dt90_ci_lower = []
    dt90_ci_upper = []
    censoring = []

    for model in models:
        with open(PSYCHO_DIR / model_files[model], 'r') as f:
            data = json.load(f)

        dt90 = data['duration_thresholds'].get('DT90', {})

        if dt90:
            dt90_values.append(dt90['point'])
            dt90_ci_lower.append(dt90['ci'][0])
            dt90_ci_upper.append(dt90['ci'][1])
            censoring.append(dt90.get('censoring', 'ok'))
        else:
            dt90_values.append(np.nan)
            dt90_ci_lower.append(np.nan)
            dt90_ci_upper.append(np.nan)
            censoring.append('undefined')

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))

    # Plot bars
    bars = ax.bar(x, dt90_values,
                   color=[COLORS[m] for m in models],
                   edgecolor='black', linewidth=1.2,
                   alpha=0.8)

    # Error bars
    errors = [
        [dt90_values[i] - dt90_ci_lower[i] if not np.isnan(dt90_values[i]) else 0 for i in range(len(models))],
        [dt90_ci_upper[i] - dt90_values[i] if not np.isnan(dt90_values[i]) else 0 for i in range(len(models))]
    ]

    ax.errorbar(x, dt90_values, yerr=errors, fmt='none',
                ecolor='black', capsize=5, capthick=1.5, linewidth=1.5)

    ax.set_ylabel('DT90 (ms)', fontweight='bold')
    ax.set_xlabel('Model Configuration', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=15, ha='right')
    ax.set_ylim([0, max([v for v in dt90_values if not np.isnan(v)]) * 1.3])

    # Add value labels
    for i, (bar, val, cens) in enumerate(zip(bars, dt90_values, censoring)):
        if not np.isnan(val):
            height = bar.get_height()
            label = f'{val:.1f}' if cens == 'ok' else f'{val:.1f}*'
            ax.text(bar.get_x() + bar.get_width()/2., height + errors[1][i] + 10,
                    label, ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax.set_title('Psychometric Thresholds: DT90 (Duration for 90% Accuracy)',
                 fontweight='bold', pad=15)
    ax.text(0.02, 0.98, 'Lower is better (more robust)',
            transform=ax.transAxes, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure3_dt90_thresholds.png')
    plt.savefig(OUTPUT_DIR / 'figure3_dt90_thresholds.pdf')
    print(f"✓ Figure 3 saved: {OUTPUT_DIR}/figure3_dt90_thresholds.png")
    plt.close()


def figure4_recall_tradeoff():
    """Figure 4: Speech vs NonSpeech recall trade-off plot."""

    configs = stats['per_config_metrics']

    models = ['baseline', 'base_opro', 'lora', 'lora_opro_classic']

    recall_speech = [configs[m]['recall_speech']['point'] for m in models]
    recall_nonspeech = [configs[m]['recall_nonspeech']['point'] for m in models]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot points
    for i, model in enumerate(models):
        ax.scatter(recall_nonspeech[i], recall_speech[i],
                   s=200, color=COLORS[model], edgecolor='black', linewidth=2,
                   label=MODEL_NAMES[model], zorder=3)

    # Diagonal (perfect balance)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Perfect balance')

    # Iso-BA lines
    for ba in [0.6, 0.7, 0.8, 0.9]:
        x = np.linspace(0, 1, 100)
        y = 2 * ba - x
        ax.plot(x, y, 'gray', alpha=0.2, linewidth=0.8)
        # Label
        if ba >= 0.6:
            ax.text(0.95, 2*ba - 0.95, f'BA={ba:.1f}',
                    fontsize=8, color='gray', ha='right', va='bottom')

    ax.set_xlabel('Recall (NonSpeech)', fontweight='bold')
    ax.set_ylabel('Recall (Speech)', fontweight='bold')
    ax.set_xlim([0.2, 1.0])
    ax.set_ylim([0.2, 1.0])
    ax.set_aspect('equal')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_title('Recall Trade-off: Speech vs NonSpeech', fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure4_recall_tradeoff.png')
    plt.savefig(OUTPUT_DIR / 'figure4_recall_tradeoff.pdf')
    print(f"✓ Figure 4 saved: {OUTPUT_DIR}/figure4_recall_tradeoff.png")
    plt.close()


def figure5_improvement_bars():
    """Figure 5: Improvement over baseline (stacked contributions)."""

    configs = stats['per_config_metrics']

    baseline_ba = configs['baseline']['ba_clip']['point']
    base_opro_ba = configs['base_opro']['ba_clip']['point']
    lora_ba = configs['lora']['ba_clip']['point']
    lora_opro_ba = configs['lora_opro_classic']['ba_clip']['point']

    # Calculate improvements
    opro_contribution = base_opro_ba - baseline_ba
    lora_contribution = lora_ba - baseline_ba
    combined_improvement = lora_opro_ba - baseline_ba

    # Breakdown of LoRA+OPRO
    lora_only = lora_ba - baseline_ba
    opro_on_lora = lora_opro_ba - lora_ba

    fig, ax = plt.subplots(figsize=(8, 5))

    models = ['Base+OPRO\nvs Baseline', 'LoRA+Hand\nvs Baseline', 'LoRA+OPRO\nvs Baseline']
    improvements = [opro_contribution, lora_contribution, combined_improvement]

    bars = ax.bar(models, improvements,
                   color=['#f39c12', '#3498db', '#2ecc71'],
                   edgecolor='black', linewidth=1.2, alpha=0.8)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'+{height:.3f}', ha='center', va='bottom',
                fontweight='bold', fontsize=10)

    # Breakdown for LoRA+OPRO
    ax.bar([2], [lora_only], bottom=0, color='#3498db',
           edgecolor='black', linewidth=1.2, alpha=0.6, label='LoRA contribution')
    ax.bar([2], [opro_on_lora], bottom=lora_only, color='#f39c12',
           edgecolor='black', linewidth=1.2, alpha=0.6, label='OPRO contribution')

    ax.set_ylabel('Improvement in BA over Baseline', fontweight='bold')
    ax.set_xlabel('Comparison', fontweight='bold')
    ax.set_ylim([0, max(improvements) * 1.2])
    ax.legend(loc='upper left')
    ax.set_title('Model Improvements over Baseline', fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure5_improvements.png')
    plt.savefig(OUTPUT_DIR / 'figure5_improvements.pdf')
    print(f"✓ Figure 5 saved: {OUTPUT_DIR}/figure5_improvements.png")
    plt.close()


if __name__ == '__main__':
    print("="*80)
    print("GENERATING PUBLICATION FIGURES")
    print("="*80)

    print("\nGenerating Figure 1: BA comparison...")
    figure1_ba_barplot()

    print("\nGenerating Figure 2: Forest plot...")
    figure2_forest_plot()

    print("\nGenerating Figure 3: Psychometric thresholds...")
    figure3_psychometric_thresholds()

    print("\nGenerating Figure 4: Recall trade-off...")
    figure4_recall_tradeoff()

    print("\nGenerating Figure 5: Improvements over baseline...")
    figure5_improvement_bars()

    print("\n" + "="*80)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"  - {file.name}")
