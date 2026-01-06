#!/usr/bin/env python3
"""
Verify data integrity across result runs.
Checks:
1. Hash consistency of predictions.csv
2. Classic vs Open: if 0 discordant pairs, all derived metrics must match
3. Consistency across all configurations in a single run
"""

import hashlib
import json
import pandas as pd
from pathlib import Path
import numpy as np
from collections import Counter

def compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def load_predictions(csv_path):
    """Load predictions CSV and return as DataFrame."""
    df = pd.read_csv(csv_path)

    # Handle different column naming schemes
    if 'clip_id' not in df.columns:
        if 'audio_path' in df.columns:
            # Extract clip_id from audio_path
            # Assuming format: .../esc50_xxx or .../voxconverse_xxx
            df['clip_id'] = df['audio_path'].apply(
                lambda x: '_'.join(Path(x).stem.split('_')[:2])
            )
        else:
            raise ValueError(f"Cannot determine clip_id from {csv_path}")

    # Determine correctness
    if 'correct' not in df.columns:
        if 'ground_truth' in df.columns and 'prediction' in df.columns:
            df['correct'] = (df['prediction'] == df['ground_truth']).astype(int)
        elif 'label' in df.columns and 'predicted_label' in df.columns:
            df['correct'] = (df['predicted_label'] == df['label']).astype(int)
        else:
            raise ValueError(f"Cannot determine correctness from {csv_path}")

    return df

def count_discordant_pairs(df_a, df_b, merge_key='clip_id'):
    """
    Count McNemar discordant pairs between two prediction sets.
    Returns: (n01, n10, total_samples)
    n01: A wrong, B correct
    n10: A correct, B wrong
    """
    # Merge on unique identifier (clip_id or audio_path)
    # Try clip_id first, then audio_path
    if merge_key in df_a.columns and merge_key in df_b.columns:
        merged = pd.merge(df_a, df_b, on=merge_key, suffixes=('_a', '_b'))
    elif 'audio_path' in df_a.columns and 'audio_path' in df_b.columns:
        merged = pd.merge(df_a, df_b, on='audio_path', suffixes=('_a', '_b'))
    else:
        raise ValueError(f"Cannot merge on {merge_key} or audio_path")

    # Use the correct column computed in load_predictions
    if 'correct_a' in merged.columns and 'correct_b' in merged.columns:
        pass  # Already have it
    else:
        raise ValueError("'correct' column not found after merge")

    n01 = ((merged['correct_a'] == 0) & (merged['correct_b'] == 1)).sum()
    n10 = ((merged['correct_a'] == 1) & (merged['correct_b'] == 0)).sum()
    total = len(merged)

    return int(n01), int(n10), int(total)

def verify_classic_vs_open(run_classic, run_open, config_pairs):
    """
    Verify Classic vs Open consistency.

    Parameters:
    -----------
    run_classic : Path
        Path to Classic run directory
    run_open : Path
        Path to Open run directory
    config_pairs : list of tuples
        List of (classic_subdir, open_subdir, config_name) to compare

    Returns:
    --------
    dict with verification results
    """
    results = {
        'consistent': True,
        'comparisons': []
    }

    for classic_sub, open_sub, config_name in config_pairs:
        classic_csv = run_classic / classic_sub / "predictions.csv"
        open_csv = run_open / open_sub / "predictions.csv"

        if not classic_csv.exists():
            print(f"❌ Missing Classic predictions: {classic_csv}")
            results['consistent'] = False
            continue
        if not open_csv.exists():
            print(f"❌ Missing Open predictions: {open_csv}")
            results['consistent'] = False
            continue

        # Compute hashes
        hash_classic = compute_file_hash(classic_csv)
        hash_open = compute_file_hash(open_csv)

        # Load and compare
        df_classic = load_predictions(classic_csv)
        df_open = load_predictions(open_csv)

        n01, n10, total = count_discordant_pairs(df_classic, df_open)

        comparison = {
            'config': config_name,
            'classic_path': str(classic_csv),
            'open_path': str(open_csv),
            'hash_classic': hash_classic,
            'hash_open': hash_open,
            'hashes_match': hash_classic == hash_open,
            'n01': n01,
            'n10': n10,
            'total_samples': total,
            'discordant_pairs': n01 + n10,
            'discordant_rate': (n01 + n10) / total if total > 0 else 0.0
        }

        # Check consistency
        if comparison['discordant_pairs'] == 0:
            # Should have identical predictions
            if not comparison['hashes_match']:
                print(f"⚠️  {config_name}: 0 discordant pairs but hashes differ!")
                print(f"    Classic: {hash_classic[:16]}...")
                print(f"    Open:    {hash_open[:16]}...")
                results['consistent'] = False
            else:
                print(f"✅ {config_name}: Identical predictions (hash match, 0 discordant)")
        else:
            print(f"ℹ️  {config_name}: {n01 + n10} discordant pairs ({comparison['discordant_rate']:.4f} rate)")
            results['consistent'] = False

        results['comparisons'].append(comparison)

    return results

def verify_run_completeness(run_dir, expected_configs):
    """
    Verify that a run directory has all expected configuration outputs.

    Parameters:
    -----------
    run_dir : Path
        Run directory
    expected_configs : list of str
        List of subdirectory names expected

    Returns:
    --------
    dict with completeness check
    """
    results = {
        'complete': True,
        'missing': [],
        'hashes': {}
    }

    for config in expected_configs:
        csv_path = run_dir / config / "predictions.csv"
        if not csv_path.exists():
            results['complete'] = False
            results['missing'].append(config)
            print(f"❌ Missing: {config}/predictions.csv")
        else:
            hash_val = compute_file_hash(csv_path)
            results['hashes'][config] = hash_val
            print(f"✅ Found: {config}/predictions.csv (hash: {hash_val[:16]}...)")

    return results

def main():
    """Main verification routine."""
    print("="*80)
    print("DATA INTEGRITY VERIFICATION")
    print("="*80)

    # Define run directories
    run_classic = Path("results/complete_pipeline_seed42")
    run_open = Path("results/complete_pipeline_seed42_opro_open")

    # Expected configs in Classic run
    classic_configs = [
        "01_baseline",
        "06_eval_base_opro",
        "03_eval_lora",
        "07_eval_lora_opro"
    ]

    # Verify Classic run completeness
    print("\n" + "─"*80)
    print("1. Verifying Classic run completeness")
    print("─"*80)
    classic_check = verify_run_completeness(run_classic, classic_configs)

    # Expected configs in Open run
    open_configs = [
        "06_eval_base_opro",
        "07_eval_lora_opro"
    ]

    # Verify Open run completeness
    print("\n" + "─"*80)
    print("2. Verifying Open run completeness")
    print("─"*80)
    open_check = verify_run_completeness(run_open, open_configs)

    # Verify Classic vs Open
    print("\n" + "─"*80)
    print("3. Verifying Classic vs Open consistency")
    print("─"*80)

    config_pairs = [
        ("06_eval_base_opro", "06_eval_base_opro", "Base+OPRO"),
        ("07_eval_lora_opro", "07_eval_lora_opro", "LoRA+OPRO")
    ]

    classic_vs_open = verify_classic_vs_open(run_classic, run_open, config_pairs)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    all_ok = (
        classic_check['complete'] and
        open_check['complete'] and
        classic_vs_open['consistent']
    )

    if all_ok:
        print("✅ ALL CHECKS PASSED")
    else:
        print("❌ INTEGRITY ISSUES DETECTED")
        if not classic_check['complete']:
            print(f"   - Classic run missing: {classic_check['missing']}")
        if not open_check['complete']:
            print(f"   - Open run missing: {open_check['missing']}")
        if not classic_vs_open['consistent']:
            print(f"   - Classic vs Open inconsistency")

    # Save verification report
    output = {
        'classic_run': str(run_classic),
        'open_run': str(run_open),
        'classic_completeness': classic_check,
        'open_completeness': open_check,
        'classic_vs_open': classic_vs_open,
        'overall_status': 'PASS' if all_ok else 'FAIL'
    }

    report_path = Path("results/data_integrity_report.json")
    with open(report_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nVerification report saved to: {report_path}")

    return 0 if all_ok else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
