#!/usr/bin/env python3
"""
Generate robustness figures by degradation axis (4-config design).
Generates Fig_R03-R06 for the paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Publication settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Configuration paths (4-config design: remove Open variant)
CONFIG_PATHS = {
    'baseline': 'results/complete_pipeline_seed42/01_baseline/predictions.csv',
    'base_opro': 'results/complete_pipeline_seed42/06_eval_base_opro/predictions.csv',
    'lora': 'results/complete_pipeline_seed42/03_eval_lora/predictions.csv',
    'lora_opro': 'results/complete_pipeline_seed42/07_eval_lora_opro/predictions.csv',
}

# Display names for configs
CONFIG_DISPLAY = {
    'baseline': 'Baseline',
    'base_opro': 'Base+OPRO',
    'lora': 'LoRA',
    'lora_opro': 'LoRA+OPRO',
}

# Colors for configs
CONFIG_COLORS = {
    'baseline': '#e74c3c',
    'base_opro': '#f39c12',
    'lora': '#3498db',
    'lora_opro': '#2ecc71',
}

# Markers for configs
CONFIG_MARKERS = {
    'baseline': 'o',
    'base_opro': 's',
    'lora': '^',
    'lora_opro': 'D',
}

OUTPUT_DIR = Path('figures')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def extract_condition_value(condition):
    """Extract numeric value from condition string."""
    # Duration: dur_20ms -> 20
    if 'dur_' in condition:
        match = re.search(r'dur_(\d+)ms', condition)
        return int(match.group(1)) if match else None

    # SNR: snr_-10dB -> -10, snr_10dB -> 10
    if 'snr_' in condition:
        match = re.search(r'snr_(-?\d+)dB', condition)
        return int(match.group(1)) if match else None

    # Reverb: reverb_0.3s -> 0.3, reverb_none -> 0
    if 'reverb_' in condition:
        if 'none' in condition:
            return 0.0
        match = re.search(r'reverb_([\d.]+)s', condition)
        return float(match.group(1)) if match else None

    # Filter: categorical
    if 'filter_' in condition:
        return condition.replace('filter_', '')

    return None


def compute_ba_by_condition(df):
    """Compute balanced accuracy by condition."""
    results = []

    # Add correct column
    df['correct'] = (df['ground_truth'] == df['prediction']).astype(int)

    for (variant_type, condition), group in df.groupby(['variant_type', 'condition']):
        # Compute recall for each class
        recall_speech = group[group['ground_truth'] == 'SPEECH']['correct'].mean()
        recall_nonspeech = group[group['ground_truth'] == 'NONSPEECH']['correct'].mean()

        # Balanced accuracy
        ba_condition = (recall_speech + recall_nonspeech) / 2

        # Extract numeric value for sorting/plotting
        condition_value = extract_condition_value(condition)

        results.append({
            'variant_type': variant_type,
            'condition': condition,
            'condition_value': condition_value,
            'BA_condition': ba_condition * 100,  # Convert to percentage
        })

    return pd.DataFrame(results)


def load_and_process_all_configs():
    """Load all configs and compute BA by condition."""
    all_results = []

    for config_name, path in CONFIG_PATHS.items():
        print(f"Processing {config_name}...")
        df = pd.read_csv(path)

        # Compute BA by condition
        ba_df = compute_ba_by_condition(df)
        ba_df['config'] = config_name

        all_results.append(ba_df)

    return pd.concat(all_results, ignore_index=True)


def plot_robustness_curve(df, variant_type, output_filename, title, xlabel):
    """Plot robustness curve for a specific variant type."""

    fig, ax = plt.subplots(figsize=(8, 6))

    # Filter for this variant type
    data = df[df['variant_type'] == variant_type].copy()

    # Sort by condition value
    data = data.sort_values('condition_value')

    # Plot each config
    for config_name in ['baseline', 'base_opro', 'lora', 'lora_opro']:
        config_data = data[data['config'] == config_name]

        if len(config_data) > 0:
            ax.plot(config_data['condition_value'], config_data['BA_condition'],
                   marker=CONFIG_MARKERS[config_name],
                   markersize=8,
                   linewidth=2,
                   color=CONFIG_COLORS[config_name],
                   label=CONFIG_DISPLAY[config_name])

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel('Balanced Accuracy (%)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{output_filename}.png")
    plt.savefig(OUTPUT_DIR / f"{output_filename}.pdf")
    print(f"  ✓ Saved: {output_filename}")
    plt.close()


if __name__ == '__main__':
    print("="*80)
    print("GENERATING ROBUSTNESS FIGURES (4-CONFIG DESIGN)")
    print("="*80)

    # Load all data
    print("\nLoading configurations...")
    df_all = load_and_process_all_configs()

    # Generate figures for each degradation axis
    print("\n[1/4] Generating Fig_R03: Duration robustness...")
    plot_robustness_curve(
        df_all,
        variant_type='duration',
        output_filename='Fig_R03_Duration_BAbyCondition',
        title='Robustness to Speech Duration',
        xlabel='Duration (ms)'
    )

    print("\n[2/4] Generating Fig_R04: SNR robustness...")
    plot_robustness_curve(
        df_all,
        variant_type='snr',
        output_filename='Fig_R04_SNR_BAbyCondition',
        title='Robustness to Signal-to-Noise Ratio',
        xlabel='SNR (dB)'
    )

    print("\n[3/4] Generating Fig_R05: Reverb robustness...")
    plot_robustness_curve(
        df_all,
        variant_type='reverb',
        output_filename='Fig_R05_Reverb_BAbyCondition',
        title='Robustness to Reverberation',
        xlabel='Reverb Time (s)'
    )

    print("\n[4/4] Generating Fig_R06: Filter robustness...")
    # For filter, we need categorical x-axis
    data_filter = df_all[df_all['variant_type'] == 'filter'].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    filter_types = sorted(data_filter['condition'].unique())
    filter_labels = [f.replace('filter_', '') for f in filter_types]
    x = np.arange(len(filter_types))
    width = 0.2

    for i, config_name in enumerate(['baseline', 'base_opro', 'lora', 'lora_opro']):
        config_data = data_filter[data_filter['config'] == config_name]

        # Match order with filter_types
        ba_values = []
        for ft in filter_types:
            row = config_data[config_data['condition'] == ft]
            if len(row) > 0:
                ba_values.append(row['BA_condition'].iloc[0])
            else:
                ba_values.append(0)

        ax.bar(x + i*width, ba_values, width,
               label=CONFIG_DISPLAY[config_name],
               color=CONFIG_COLORS[config_name],
               edgecolor='black',
               linewidth=1)

    ax.set_xlabel('Filter Type', fontweight='bold')
    ax.set_ylabel('Balanced Accuracy (%)', fontweight='bold')
    ax.set_title('Robustness to Frequency Filtering', fontweight='bold', pad=15)
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(filter_labels, rotation=15, ha='right')
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Fig_R06_Filter_BAbyCondition.png')
    plt.savefig(OUTPUT_DIR / 'Fig_R06_Filter_BAbyCondition.pdf')
    print(f"  ✓ Saved: Fig_R06_Filter_BAbyCondition")
    plt.close()

    print("\n" + "="*80)
    print("ROBUSTNESS FIGURES GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
