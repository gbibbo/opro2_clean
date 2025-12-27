#!/usr/bin/env python3
"""
Compute Psychometric Thresholds

Calculates DT50, DT75, DT90, and SNR-75 with bootstrap CIs for
specified model configurations.

Imports and uses functions from statistical_analysis.py to ensure
consistency with the paper's methodology.

Usage:
    # Process all configurations
    python compute_psychometric.py --all

    # Process specific configurations
    python compute_psychometric.py --configs baseline lora_hand

    # Process only OPRO varied configs
    python compute_psychometric.py --configs base_opro_varied lora_opro_varied
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

# Import functions from statistical_analysis.py
from statistical_analysis import (
    load_predictions,
    cluster_bootstrap_thresholds,
    compute_ba,
    cluster_bootstrap_ba,
    compute_recalls_with_wilson
)

# Configuration paths
RESULTS_DIR = Path('/mnt/fast/nobackup/users/gb0048/opro2_clean/results')

CONFIGURATIONS = {
    'baseline': {
        'name': 'Baseline (Hand-crafted)',
        'path': RESULTS_DIR / 'complete_pipeline_seed42/01_baseline/predictions.csv',
        'type': 'baseline'
    },
    'lora_hand': {
        'name': 'LoRA + Hand-crafted',
        'path': RESULTS_DIR / 'complete_pipeline_seed42/03_eval_lora/predictions.csv',
        'type': 'lora'
    },
    'base_opro_classic': {
        'name': 'Base + OPRO (Classic)',
        'path': RESULTS_DIR / 'complete_pipeline_seed42/06_eval_base_opro/predictions.csv',
        'type': 'base_opro'
    },
    'lora_opro_classic': {
        'name': 'LoRA + OPRO (Classic)',
        'path': RESULTS_DIR / 'complete_pipeline_seed42/07_eval_lora_opro/predictions.csv',
        'type': 'lora_opro'
    },
    'base_opro_open': {
        'name': 'Base + OPRO (Open)',
        'path': RESULTS_DIR / 'complete_pipeline_seed42_opro_open/06_eval_base_opro/predictions.csv',
        'type': 'base_opro_open'
    },
    'lora_opro_open': {
        'name': 'LoRA + OPRO (Open)',
        'path': RESULTS_DIR / 'complete_pipeline_seed42_opro_open/07_eval_lora_opro/predictions.csv',
        'type': 'lora_opro_open'
    },
    'base_opro_varied': {
        'name': 'Base + OPRO (Varied)',
        'path': RESULTS_DIR / 'opro_varied_seed42/eval_base/predictions.csv',
        'type': 'base_opro_varied'
    },
    'lora_opro_varied': {
        'name': 'LoRA + OPRO (Varied)',
        'path': RESULTS_DIR / 'opro_varied_seed42/eval_lora/predictions.csv',
        'type': 'lora_opro_varied'
    },
}


def compute_all_metrics(df, config_name, n_bootstrap=10000, random_state=42):
    """Compute all metrics for a configuration."""
    print(f"\n{'='*80}")
    print(f"Computing metrics for: {config_name}")
    print(f"{'='*80}")

    results = {}

    # 1. Balanced accuracy with bootstrap CI
    print("  Computing BA with cluster bootstrap...")
    ba, ba_lower, ba_upper = cluster_bootstrap_ba(
        df, n_bootstrap=n_bootstrap, random_state=random_state
    )
    results['ba_clip'] = {
        'point': float(ba),
        'ci': [float(ba_lower), float(ba_upper)]
    }
    print(f"    BA_clip: {ba:.4f} [{ba_lower:.4f}, {ba_upper:.4f}]")

    # 2. Per-class recalls with Wilson score CIs
    print("  Computing per-class recalls...")
    recalls = compute_recalls_with_wilson(df)
    results['recalls'] = recalls
    print(f"    Recall_Speech: {recalls['recall_speech']:.4f} "
          f"[{recalls['recall_speech_ci'][0]:.4f}, {recalls['recall_speech_ci'][1]:.4f}]")
    print(f"    Recall_NonSpeech: {recalls['recall_nonspeech']:.4f} "
          f"[{recalls['recall_nonspeech_ci'][0]:.4f}, {recalls['recall_nonspeech_ci'][1]:.4f}]")

    # 3. Duration thresholds (DT50, DT75, DT90)
    print("  Computing duration thresholds (DT50/75/90)...")
    dt_thresholds = cluster_bootstrap_thresholds(
        df, 'duration', targets=[0.50, 0.75, 0.90],
        n_bootstrap=n_bootstrap, random_state=random_state
    )
    results['duration_thresholds'] = dt_thresholds

    for key, val in dt_thresholds.items():
        print(f"    {key}: {val['point']:.2f} ms "
              f"[{val['ci'][0]:.2f}, {val['ci'][1]:.2f}]")

    # 4. SNR threshold (SNR-75)
    print("  Computing SNR threshold (SNR-75)...")
    snr_thresholds = cluster_bootstrap_thresholds(
        df, 'snr', targets=[0.75],
        n_bootstrap=n_bootstrap, random_state=random_state
    )
    results['snr_thresholds'] = snr_thresholds

    for key, val in snr_thresholds.items():
        print(f"    {key}: {val['point']:.2f} dB "
              f"[{val['ci'][0]:.2f}, {val['ci'][1]:.2f}]")

    # 5. Sample sizes
    results['n_samples'] = len(df)
    results['n_clips'] = df['clip_id'].nunique()

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute psychometric thresholds for model configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all configurations
  %(prog)s --all

  # Process specific configurations
  %(prog)s --configs baseline lora_hand

  # Process only varied seed OPRO configs
  %(prog)s --configs base_opro_varied lora_opro_varied

Available configurations:
  baseline, lora_hand, base_opro_classic, lora_opro_classic,
  base_opro_open, lora_opro_open, base_opro_varied, lora_opro_varied
"""
    )

    # Configuration selection (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '--all',
        action='store_true',
        help='Process all configurations'
    )
    config_group.add_argument(
        '--configs',
        nargs='+',
        choices=list(CONFIGURATIONS.keys()),
        help='Specific configurations to process'
    )

    # Other arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(RESULTS_DIR / 'psychometric_analysis'),
        help='Output directory (default: results/psychometric_analysis)'
    )
    parser.add_argument(
        '--n_bootstrap',
        type=int,
        default=10000,
        help='Number of bootstrap samples (default: 10000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    # Determine which configs to process
    if args.all:
        configs_to_process = list(CONFIGURATIONS.keys())
    else:
        configs_to_process = args.configs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("PSYCHOMETRIC THRESHOLDS")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"Random seed: {args.seed}")
    print(f"Configurations to process: {len(configs_to_process)}")

    # Process selected configurations
    all_results = {}
    skipped_configs = []

    for config_key in configs_to_process:
        config_info = CONFIGURATIONS[config_key]
        config_name = config_info['name']
        csv_path = config_info['path']

        if not csv_path.exists():
            print(f"\n⚠️  Skipping {config_name}: file not found")
            print(f"    {csv_path}")
            skipped_configs.append(config_key)
            continue

        # Load data
        print(f"\nLoading {config_name}...")
        df = load_predictions(str(csv_path))
        print(f"  Loaded {len(df)} samples ({df['clip_id'].nunique()} unique clips)")

        # Compute metrics
        results = compute_all_metrics(
            df, config_name,
            n_bootstrap=args.n_bootstrap,
            random_state=args.seed
        )

        # Add metadata
        results['config_key'] = config_key
        results['config_name'] = config_name
        results['config_type'] = config_info['type']
        results['csv_path'] = str(csv_path)

        all_results[config_key] = results

        # Save individual results
        individual_file = output_dir / f"{config_key}_psychometric.json"
        with open(individual_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved to: {individual_file}")

    # Save combined results
    if all_results:
        combined_file = output_dir / 'all_psychometric_thresholds.json'
        with open(combined_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'n_bootstrap': args.n_bootstrap,
                'random_seed': args.seed,
                'configurations': all_results
            }, f, indent=2)

        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Configurations processed: {len(all_results)}")
        if skipped_configs:
            print(f"Configurations skipped: {len(skipped_configs)} ({', '.join(skipped_configs)})")
        print(f"Results saved to: {output_dir}")

        # Create text report
        report_file = output_dir / 'psychometric_report.txt'
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PSYCHOMETRIC THRESHOLDS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Bootstrap samples: {args.n_bootstrap}\n")
            f.write(f"Random seed: {args.seed}\n")
            f.write(f"Configurations: {len(all_results)}\n\n")

            for config_key, results in all_results.items():
                f.write("="*80 + "\n")
                f.write(f"{results['config_name']}\n")
                f.write("="*80 + "\n")
                f.write(f"Type: {results['config_type']}\n")
                f.write(f"Samples: {results['n_samples']} ({results['n_clips']} clips)\n\n")

                # BA
                ba = results['ba_clip']
                f.write(f"Balanced Accuracy:\n")
                f.write(f"  BA_clip: {ba['point']:.4f} [{ba['ci'][0]:.4f}, {ba['ci'][1]:.4f}]\n\n")

                # Recalls
                recalls = results['recalls']
                f.write(f"Per-class Recalls:\n")
                f.write(f"  Speech: {recalls['recall_speech']:.4f} "
                       f"[{recalls['recall_speech_ci'][0]:.4f}, {recalls['recall_speech_ci'][1]:.4f}] "
                       f"(n={recalls['n_speech']})\n")
                f.write(f"  NonSpeech: {recalls['recall_nonspeech']:.4f} "
                       f"[{recalls['recall_nonspeech_ci'][0]:.4f}, {recalls['recall_nonspeech_ci'][1]:.4f}] "
                       f"(n={recalls['n_nonspeech']})\n\n")

                # Duration thresholds
                f.write(f"Duration Thresholds:\n")
                if results['duration_thresholds']:
                    for key, val in results['duration_thresholds'].items():
                        f.write(f"  {key}: {val['point']:.2f} ms "
                               f"[{val['ci'][0]:.2f}, {val['ci'][1]:.2f}]\n")
                else:
                    f.write("  (None computed - accuracy does not cross thresholds)\n")
                f.write("\n")

                # SNR thresholds
                f.write(f"SNR Thresholds:\n")
                if results['snr_thresholds']:
                    for key, val in results['snr_thresholds'].items():
                        f.write(f"  {key}: {val['point']:.2f} dB "
                               f"[{val['ci'][0]:.2f}, {val['ci'][1]:.2f}]\n")
                else:
                    f.write("  (None computed - accuracy does not cross threshold)\n")
                f.write("\n")

        print(f"Text report saved to: {report_file}")

    print("\n" + "="*80)
    print("DONE")
    print("="*80)

    return 0 if all_results else 1


if __name__ == '__main__':
    sys.exit(main())
