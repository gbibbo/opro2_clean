#!/usr/bin/env python3
"""
Generate robustness figures by degradation axis (condition-level results).
Results.4: Condition-Level Robustness Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Configuration paths (in desired order)
CONFIG_PATHS = {
    'baseline': 'results/complete_pipeline_seed42/01_baseline/predictions.csv',
    'base_opro_classic': 'results/complete_pipeline_seed42/06_eval_base_opro/predictions.csv',
    'lora_hand': 'results/complete_pipeline_seed42/03_eval_lora/predictions.csv',
    'lora_opro_classic': 'results/complete_pipeline_seed42/07_eval_lora_opro/predictions.csv',
    'lora_opro_open': 'results/complete_pipeline_seed42_opro_open/07_eval_lora_opro/predictions.csv',
}

# Display names for configs
CONFIG_DISPLAY = {
    'baseline': 'Baseline',
    'base_opro_classic': 'Base + OPRO',
    'lora_hand': 'LoRA Hand',
    'lora_opro_classic': 'LoRA + OPRO',
    'lora_opro_open': 'LoRA + OPRO-Open',
}

# Colors for configs (consistent across figures)
CONFIG_COLORS = {
    'baseline': '#1f77b4',
    'base_opro_classic': '#ff7f0e',
    'lora_hand': '#2ca02c',
    'lora_opro_classic': '#d62728',
    'lora_opro_open': '#9467bd',
}

# Markers for configs
CONFIG_MARKERS = {
    'baseline': 'o',
    'base_opro_classic': 's',
    'lora_hand': '^',
    'lora_opro_classic': 'D',
    'lora_opro_open': 'v',
}


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
            'BA_condition': ba_condition,
            'n_total': len(group),
            'n_speech': len(group[group['ground_truth'] == 'SPEECH']),
            'n_nonspeech': len(group[group['ground_truth'] == 'NONSPEECH']),
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

    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)

    # Reorder columns
    combined = combined[['config', 'variant_type', 'condition', 'condition_value',
                         'BA_condition', 'n_total', 'n_speech', 'n_nonspeech']]

    return combined


def plot_robustness_by_axis(df, variant_type, output_prefix, xlabel, ylabel='Balanced Accuracy'):
    """Plot robustness for a specific degradation axis."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Filter to variant type
    df_variant = df[df['variant_type'] == variant_type].copy()

    # Plot each config
    for config_name in CONFIG_PATHS.keys():
        df_config = df_variant[df_variant['config'] == config_name].sort_values('condition_value')

        if len(df_config) > 0:
            ax.plot(df_config['condition_value'], df_config['BA_condition'],
                   marker=CONFIG_MARKERS[config_name],
                   color=CONFIG_COLORS[config_name],
                   label=CONFIG_DISPLAY[config_name],
                   linewidth=2,
                   markersize=8)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')

    plt.tight_layout()

    # Save both PDF and PNG
    fig.savefig(f'paper_artifacts/figures/{output_prefix}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'paper_artifacts/figures/{output_prefix}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved {output_prefix}.pdf and .png")


def main():
    # Load and process all configs
    print("Loading and processing all configs...")
    df_all = load_and_process_all_configs()

    # Save combined data
    output_csv = 'paper_artifacts/data/BA_byCondition_AllConfigs.csv'
    df_all.to_csv(output_csv, index=False)
    print(f"\nSaved {output_csv}")
    print(f"Total rows: {len(df_all)}")

    # Generate figures for each axis
    print("\nGenerating figures...")

    # Duration
    plot_robustness_by_axis(
        df_all,
        variant_type='duration',
        output_prefix='Fig_R03_Duration_BAbyCondition',
        xlabel='Duration (ms)'
    )

    # SNR
    plot_robustness_by_axis(
        df_all,
        variant_type='snr',
        output_prefix='Fig_R04_SNR_BAbyCondition',
        xlabel='SNR (dB)'
    )

    # Reverb
    plot_robustness_by_axis(
        df_all,
        variant_type='reverb',
        output_prefix='Fig_R05_Reverb_BAbyCondition',
        xlabel='Reverberation Time RT60 (s)'
    )

    # Filter - handle categorical
    df_filter = df_all[df_all['variant_type'] == 'filter'].copy()

    # Define filter order
    filter_order = ['none', 'bandpass', 'lowpass', 'highpass']
    filter_order_map = {f: i for i, f in enumerate(filter_order)}
    df_filter['filter_order'] = df_filter['condition_value'].map(filter_order_map)

    fig, ax = plt.subplots(figsize=(8, 6))

    for config_name in CONFIG_PATHS.keys():
        df_config = df_filter[df_filter['config'] == config_name].sort_values('filter_order')

        if len(df_config) > 0:
            ax.plot(df_config['filter_order'], df_config['BA_condition'],
                   marker=CONFIG_MARKERS[config_name],
                   color=CONFIG_COLORS[config_name],
                   label=CONFIG_DISPLAY[config_name],
                   linewidth=2,
                   markersize=8)

    ax.set_xlabel('Filter Type', fontsize=12)
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_xticks(range(len(filter_order)))
    ax.set_xticklabels(filter_order)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    fig.savefig('paper_artifacts/figures/Fig_R06_Filter_BAbyCondition.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('paper_artifacts/figures/Fig_R06_Filter_BAbyCondition.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved Fig_R06_Filter_BAbyCondition.pdf and .png")

    print("\n✓ All robustness figures generated successfully!")

    # Summary stats
    print("\n=== Summary Statistics ===")
    print(f"Configs analyzed: {df_all['config'].nunique()}")
    print(f"Variant types: {df_all['variant_type'].nunique()}")
    print(f"Total conditions: {df_all['condition'].nunique()}")
    print("\nConditions per variant type:")
    print(df_all.groupby('variant_type')['condition'].nunique())


if __name__ == '__main__':
    main()
