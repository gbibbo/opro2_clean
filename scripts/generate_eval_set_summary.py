#!/usr/bin/env python3
"""
Generate Evaluation Set Summary for Paper (Results.1)

Extracts evaluation set statistics from actual predictions.csv files:
- Sample and clip counts
- Class balance (SPEECH/NONSPEECH)
- Condition/variant distribution
- Balance verification by condition

Outputs:
- paper_artifacts/tables/Tab_R01_EvalSetSummary.csv
- paper_artifacts/tables/Tab_R01_EvalSetSummary.tex
- paper_artifacts/data/EvalSet_ClassBalance_ByCondition.csv
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# Add scripts to path to import from statistical_analysis
sys.path.insert(0, str(Path(__file__).parent))

from statistical_analysis import load_predictions, extract_clip_id


def extract_condition_and_variant(audio_path: str) -> tuple:
    """
    Extract condition and variant_type from audio path.

    Args:
        audio_path: Full path to audio file

    Returns:
        (condition, variant_type) tuple

    Example:
        'esc50_1-45645-A-31_0152_1000ms_dur20ms.wav' -> ('dur_20ms', 'duration')
        'esc50_1-45645-A-31_0152_1000ms_snr-10dB.wav' -> ('snr_-10dB', 'snr')
    """
    filename = os.path.basename(audio_path).replace('.wav', '')

    # Detect variant type by searching for indicators
    if 'dur' in filename.lower():
        variant_type = 'duration'
        # Extract duration value (e.g., dur20ms, dur_20ms)
        match = re.search(r'dur_?(\d+(?:\.\d+)?)ms', filename, re.IGNORECASE)
        if match:
            condition = f"dur_{match.group(1)}ms"
        else:
            condition = 'duration_unknown'

    elif 'snr' in filename.lower():
        variant_type = 'snr'
        # Extract SNR value (e.g., snr-10dB, snr_-10dB, snr10dB)
        match = re.search(r'snr_?([+-]?\d+(?:\.\d+)?)d[bB]', filename, re.IGNORECASE)
        if match:
            condition = f"snr_{match.group(1)}dB"
        else:
            condition = 'snr_unknown'

    elif 'reverb' in filename.lower():
        variant_type = 'reverb'
        # Extract reverb value
        match = re.search(r'reverb_?(\w+)', filename, re.IGNORECASE)
        if match:
            condition = f"reverb_{match.group(1)}"
        else:
            condition = 'reverb_unknown'

    elif 'filter' in filename.lower():
        variant_type = 'filter'
        # Extract filter value
        match = re.search(r'filter_?(\w+)', filename, re.IGNORECASE)
        if match:
            condition = f"filter_{match.group(1)}"
        else:
            condition = 'filter_unknown'
    else:
        variant_type = 'unknown'
        condition = 'unknown'

    return condition, variant_type


def analyze_eval_set(csv_path: str, config_name: str = 'baseline') -> dict:
    """
    Analyze evaluation set from predictions.csv.

    Args:
        csv_path: Path to predictions.csv
        config_name: Name of configuration (for reporting)

    Returns:
        Dictionary with evaluation set statistics
    """
    print(f"\n{'='*80}")
    print(f"Analyzing evaluation set: {config_name}")
    print(f"{'='*80}")
    print(f"Source: {csv_path}")

    # Load predictions (already handles canonicalization and clip_id extraction)
    df = load_predictions(csv_path)

    # Add condition and variant_type
    df[['condition', 'variant_type']] = df['audio_path'].apply(
        lambda x: pd.Series(extract_condition_and_variant(x))
    )

    # Basic counts
    n_samples = len(df)
    n_clips = df['clip_id'].nunique()

    print(f"\nBasic Statistics:")
    print(f"  Total samples: {n_samples:,}")
    print(f"  Unique clips: {n_clips:,}")
    print(f"  Samples per clip: {n_samples / n_clips:.2f}")

    # Class balance (global)
    class_counts = df['ground_truth'].value_counts()
    n_speech = class_counts.get('SPEECH', 0)
    n_nonspeech = class_counts.get('NONSPEECH', 0)
    prop_speech = n_speech / n_samples if n_samples > 0 else 0
    prop_nonspeech = n_nonspeech / n_samples if n_samples > 0 else 0

    print(f"\nGlobal Class Balance:")
    print(f"  SPEECH: {n_speech:,} ({prop_speech:.2%})")
    print(f"  NONSPEECH: {n_nonspeech:,} ({prop_nonspeech:.2%})")

    # Check if balanced (within 1% tolerance)
    is_balanced = abs(prop_speech - 0.5) < 0.01
    print(f"  Balanced: {'✓ Yes' if is_balanced else '✗ No'}")

    # Variant type distribution
    print(f"\nVariant Type Distribution:")
    variant_counts = df['variant_type'].value_counts().sort_index()
    for variant, count in variant_counts.items():
        print(f"  {variant}: {count:,} ({count/n_samples:.2%})")

    # Condition distribution
    print(f"\nCondition Distribution (top 20):")
    condition_counts = df['condition'].value_counts().head(20)
    for condition, count in condition_counts.items():
        print(f"  {condition}: {count:,}")

    # Class balance by condition and variant_type
    print(f"\nClass Balance by Condition/Variant:")
    balance_by_condition = []

    for (condition, variant_type), group in df.groupby(['condition', 'variant_type']):
        n_total = len(group)
        n_speech_cond = (group['ground_truth'] == 'SPEECH').sum()
        n_nonspeech_cond = (group['ground_truth'] == 'NONSPEECH').sum()

        balance_by_condition.append({
            'condition': condition,
            'variant_type': variant_type,
            'n_total': n_total,
            'n_speech': n_speech_cond,
            'n_nonspeech': n_nonspeech_cond,
            'prop_speech': n_speech_cond / n_total if n_total > 0 else 0,
            'prop_nonspeech': n_nonspeech_cond / n_total if n_total > 0 else 0
        })

    balance_df = pd.DataFrame(balance_by_condition)

    # Check for imbalanced conditions (outside 45%-55% range)
    imbalanced = balance_df[
        (balance_df['prop_speech'] < 0.45) | (balance_df['prop_speech'] > 0.55)
    ]

    if len(imbalanced) > 0:
        print(f"  ⚠️  Found {len(imbalanced)} imbalanced conditions (outside 45%-55%):")
        for _, row in imbalanced.head(10).iterrows():
            print(f"    {row['condition']}: {row['prop_speech']:.2%} SPEECH")
    else:
        print(f"  ✓ All conditions balanced within 45%-55%")

    # Summary statistics
    summary = {
        'config_name': config_name,
        'n_samples': n_samples,
        'n_clips': n_clips,
        'samples_per_clip': n_samples / n_clips if n_clips > 0 else 0,
        'n_speech': n_speech,
        'n_nonspeech': n_nonspeech,
        'prop_speech': prop_speech,
        'prop_nonspeech': prop_nonspeech,
        'is_balanced': is_balanced,
        'n_variant_types': len(variant_counts),
        'n_conditions': len(condition_counts),
        'n_imbalanced_conditions': len(imbalanced)
    }

    return summary, balance_df


def export_summary_table(summary: dict, output_dir: Path):
    """
    Export summary statistics as CSV and LaTeX table.

    Args:
        summary: Dictionary with summary statistics
        output_dir: Output directory (paper_artifacts/tables/)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create summary DataFrame (single row)
    summary_df = pd.DataFrame([{
        'Metric': 'Total Samples',
        'Value': f"{summary['n_samples']:,}"
    }, {
        'Metric': 'Unique Clips',
        'Value': f"{summary['n_clips']:,}"
    }, {
        'Metric': 'Samples per Clip',
        'Value': f"{summary['samples_per_clip']:.2f}"
    }, {
        'Metric': 'SPEECH Samples',
        'Value': f"{summary['n_speech']:,} ({summary['prop_speech']:.1%})"
    }, {
        'Metric': 'NONSPEECH Samples',
        'Value': f"{summary['n_nonspeech']:,} ({summary['prop_nonspeech']:.1%})"
    }, {
        'Metric': 'Globally Balanced',
        'Value': 'Yes' if summary['is_balanced'] else 'No'
    }, {
        'Metric': 'Number of Variant Types',
        'Value': f"{summary['n_variant_types']}"
    }, {
        'Metric': 'Number of Conditions',
        'Value': f"{summary['n_conditions']}"
    }])

    # Export CSV
    csv_path = output_dir / 'Tab_R01_EvalSetSummary.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"\n✓ Exported CSV: {csv_path}")

    # Export LaTeX (booktabs style)
    latex_path = output_dir / 'Tab_R01_EvalSetSummary.tex'

    with open(latex_path, 'w') as f:
        f.write("% Evaluation Set Summary Table\n")
        f.write("% Auto-generated - do not edit manually\n\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Evaluation Set Summary Statistics}\n")
        f.write("\\label{tab:eval_set_summary}\n")
        f.write("\\begin{tabular}{lr}\n")
        f.write("\\toprule\n")
        f.write("Metric & Value \\\\\n")
        f.write("\\midrule\n")

        for _, row in summary_df.iterrows():
            metric = row['Metric'].replace('_', ' ')
            value = row['Value']
            f.write(f"{metric} & {value} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"✓ Exported LaTeX: {latex_path}")


def export_balance_by_condition(balance_df: pd.DataFrame, output_dir: Path):
    """
    Export class balance by condition to CSV.

    Args:
        balance_df: DataFrame with per-condition class balance
        output_dir: Output directory (paper_artifacts/data/)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by variant_type and condition
    balance_df = balance_df.sort_values(['variant_type', 'condition'])

    # Export
    csv_path = output_dir / 'EvalSet_ClassBalance_ByCondition.csv'
    balance_df.to_csv(csv_path, index=False)
    print(f"✓ Exported condition-level balance: {csv_path}")

    # Print summary statistics
    print(f"\nCondition-level balance statistics:")
    print(f"  Mean SPEECH proportion: {balance_df['prop_speech'].mean():.2%}")
    print(f"  Std SPEECH proportion: {balance_df['prop_speech'].std():.2%}")
    print(f"  Min SPEECH proportion: {balance_df['prop_speech'].min():.2%}")
    print(f"  Max SPEECH proportion: {balance_df['prop_speech'].max():.2%}")


def main():
    # Configuration paths (from compute_psychometric.py)
    RESULTS_DIR = Path('/mnt/fast/nobackup/users/gb0048/opro2_clean/results')

    CONFIGURATIONS = {
        'baseline': RESULTS_DIR / 'complete_pipeline_seed42/01_baseline/predictions.csv',
        'base_opro_classic': RESULTS_DIR / 'complete_pipeline_seed42/06_eval_base_opro/predictions.csv',
        'lora_hand': RESULTS_DIR / 'complete_pipeline_seed42/03_eval_lora/predictions.csv',
        'lora_opro_classic': RESULTS_DIR / 'complete_pipeline_seed42/07_eval_lora_opro/predictions.csv',
        'lora_opro_open': RESULTS_DIR / 'complete_pipeline_seed42_opro_open/07_eval_lora_opro/predictions.csv',
    }

    # Output directory
    OUTPUT_ROOT = Path('/mnt/fast/nobackup/users/gb0048/opro2_clean/paper_artifacts')

    print("="*80)
    print("EVALUATION SET SUMMARY EXTRACTION (Results.1)")
    print("="*80)

    # Analyze baseline (primary source of truth)
    baseline_path = CONFIGURATIONS['baseline']

    if not baseline_path.exists():
        print(f"\n❌ ERROR: Baseline predictions not found: {baseline_path}")
        return 1

    summary, balance_df = analyze_eval_set(str(baseline_path), 'baseline')

    # Verify with another config (optional - check data consistency)
    print(f"\n{'='*80}")
    print("Verifying with lora_hand configuration...")
    print(f"{'='*80}")

    lora_path = CONFIGURATIONS['lora_hand']
    if lora_path.exists():
        summary_lora, balance_lora = analyze_eval_set(str(lora_path), 'lora_hand')

        # Check if datasets match
        if summary['n_samples'] != summary_lora['n_samples']:
            print(f"\n⚠️  WARNING: Sample counts differ!")
            print(f"  baseline: {summary['n_samples']:,}")
            print(f"  lora_hand: {summary_lora['n_samples']:,}")
        else:
            print(f"\n✓ Verification passed: Both configs have {summary['n_samples']:,} samples")

    # Export artifacts
    print(f"\n{'='*80}")
    print("Exporting artifacts...")
    print(f"{'='*80}")

    export_summary_table(summary, OUTPUT_ROOT / 'tables')
    export_balance_by_condition(balance_df, OUTPUT_ROOT / 'data')

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")
    print(f"\nArtifacts exported to: {OUTPUT_ROOT}")
    print(f"  Tables: {OUTPUT_ROOT / 'tables'}")
    print(f"  Data: {OUTPUT_ROOT / 'data'}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
