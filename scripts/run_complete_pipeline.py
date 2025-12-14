#!/usr/bin/env python3
"""
Complete Pipeline Wrapper
Executes all 7 stages of the OPRO+LoRA pipeline sequentially.

Usage:
    python scripts/run_complete_pipeline.py --seed 42 --data_root "c:/VS projects/opro2/data"
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd, description, dry_run=False):
    """
    Execute a shell command and handle errors.

    Args:
        cmd: Command to execute (list or string)
        description: Human-readable description of the command
        dry_run: If True, print command without executing
    """
    print("\n" + "=" * 80)
    print(f"STAGE: {description}")
    print("=" * 80)

    if isinstance(cmd, list):
        cmd_str = " ".join(cmd)
    else:
        cmd_str = cmd

    print(f"Command: {cmd_str}\n")

    if dry_run:
        print("[DRY RUN] Would execute the above command")
        return True

    try:
        result = subprocess.run(
            cmd_str,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print(result.stdout)
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error:")
        print(e.stdout)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run complete 7-stage OPRO+LoRA pipeline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_root", type=str, default="c:/VS projects/opro2/data",
                       help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: results/pipeline_run_YYYYMMDD_HHMMSS)")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip LoRA training (use existing checkpoint)")
    parser.add_argument("--skip_opro_base", action="store_true",
                       help="Skip OPRO on base model (use existing prompt)")
    parser.add_argument("--skip_opro_lora", action="store_true",
                       help="Skip OPRO on LoRA model (use existing prompt)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print commands without executing")
    args = parser.parse_args()

    # Set up output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/pipeline_run_{timestamp}"
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Define paths
    manifest = f"{args.data_root}/processed/conditions_final/conditions_manifest_split.parquet"
    checkpoint_dir = f"checkpoints/qwen_lora_seed{args.seed}"
    checkpoint_path = f"{checkpoint_dir}/final"

    print("=" * 80)
    print("OPRO+LoRA COMPLETE PIPELINE")
    print("=" * 80)
    print(f"Seed: {args.seed}")
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 80)

    # Track failed stages
    failed_stages = []

    # =========================================================================
    # STAGE 1: Baseline Evaluation (BASE model, default prompt)
    # =========================================================================
    baseline_prompt = "Does this audio contain human speech? Answer SPEECH or NONSPEECH."
    baseline_out = f"{output_dir}/01_eval_base_baseline.csv"

    cmd = f"""python3 scripts/evaluate_simple.py \\
        --manifest "{manifest}" \\
        --prompt "{baseline_prompt}" \\
        --output_dir "{output_dir}/01_baseline" \\
        --batch_size 50"""

    if not run_command(cmd, "Stage 1: Baseline Evaluation", args.dry_run):
        failed_stages.append("Stage 1")

    # =========================================================================
    # STAGE 2: LoRA Fine-tuning
    # =========================================================================
    if not args.skip_training:
        train_csv = f"{args.data_root}/processed/experimental_variants/train_metadata.csv"
        val_csv = f"{args.data_root}/processed/experimental_variants/test_metadata.csv"

        cmd = f"""python3 scripts/finetune_qwen_audio.py \\
            --train_csv "{train_csv}" \\
            --val_csv "{val_csv}" \\
            --output_dir "{checkpoint_dir}" \\
            --seed {args.seed} \\
            --num_epochs 3 \\
            --lora_r 64 \\
            --lora_alpha 16"""

        if not run_command(cmd, "Stage 2: LoRA Fine-tuning", args.dry_run):
            failed_stages.append("Stage 2")
            print("\n⚠️ WARNING: Training failed. Subsequent stages may fail.")
    else:
        print(f"\n⏭️ Skipping Stage 2 (using existing checkpoint: {checkpoint_path})")

    # =========================================================================
    # STAGE 3: Evaluation BASE vs LoRA (both with baseline prompt)
    # =========================================================================
    # 3a: BASE model
    cmd = f"""python3 scripts/evaluate_simple.py \\
        --manifest "{manifest}" \\
        --prompt "{baseline_prompt}" \\
        --output_dir "{output_dir}/03_eval_base" \\
        --batch_size 50"""

    if not run_command(cmd, "Stage 3a: Evaluate BASE model", args.dry_run):
        failed_stages.append("Stage 3a")

    # 3b: LoRA model
    cmd = f"""python3 scripts/evaluate_simple.py \\
        --manifest "{manifest}" \\
        --prompt "{baseline_prompt}" \\
        --output_dir "{output_dir}/03_eval_lora" \\
        --checkpoint "{checkpoint_path}" \\
        --batch_size 50"""

    if not run_command(cmd, "Stage 3b: Evaluate LoRA model", args.dry_run):
        failed_stages.append("Stage 3b")

    # =========================================================================
    # STAGE 4: OPRO on BASE model
    # =========================================================================
    if not args.skip_opro_base:
        cmd = f"""python3 scripts/opro_classic_optimize.py \\
            --manifest "{manifest}" \\
            --split dev \\
            --output_dir "{output_dir}/04_opro_base" \\
            --no_lora \\
            --num_iterations 30 \\
            --candidates_per_iter 3 \\
            --seed {args.seed}"""

        if not run_command(cmd, "Stage 4: OPRO on BASE model", args.dry_run):
            failed_stages.append("Stage 4")
    else:
        print(f"\n⏭️ Skipping Stage 4 (using existing BASE prompt)")

    # =========================================================================
    # STAGE 5: OPRO on LoRA model
    # =========================================================================
    if not args.skip_opro_lora:
        dev_csv = f"{args.data_root}/processed/experimental_variants/dev_metadata.csv"

        cmd = f"""python3 scripts/opro_post_ft_v2.py \\
            --checkpoint "{checkpoint_path}" \\
            --train_csv "{dev_csv}" \\
            --output_dir "{output_dir}/05_opro_lora" \\
            --num_iterations 15 \\
            --samples_per_iter 20"""

        if not run_command(cmd, "Stage 5: OPRO on LoRA model", args.dry_run):
            failed_stages.append("Stage 5")
    else:
        print(f"\n⏭️ Skipping Stage 5 (using existing LoRA prompt)")

    # =========================================================================
    # STAGE 6: Evaluate BASE + OPRO
    # =========================================================================
    best_prompt_base = f"{output_dir}/04_opro_base/best_prompt.txt"

    cmd = f"""python3 scripts/evaluate_simple.py \\
        --manifest "{manifest}" \\
        --prompt_file "{best_prompt_base}" \\
        --output_dir "{output_dir}/06_eval_base_opro" \\
        --batch_size 50"""

    if not run_command(cmd, "Stage 6: Evaluate BASE + OPRO", args.dry_run):
        failed_stages.append("Stage 6")

    # =========================================================================
    # STAGE 7: Evaluate LoRA + OPRO (BEST)
    # =========================================================================
    best_prompt_lora = f"{output_dir}/05_opro_lora/best_prompt.txt"

    cmd = f"""python3 scripts/evaluate_simple.py \\
        --manifest "{manifest}" \\
        --prompt_file "{best_prompt_lora}" \\
        --output_dir "{output_dir}/07_eval_lora_opro" \\
        --checkpoint "{checkpoint_path}" \\
        --batch_size 50"""

    if not run_command(cmd, "Stage 7: Evaluate LoRA + OPRO", args.dry_run):
        failed_stages.append("Stage 7")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)

    if not failed_stages:
        print("✓ All stages completed successfully!")
    else:
        print(f"✗ {len(failed_stages)} stage(s) failed:")
        for stage in failed_stages:
            print(f"  - {stage}")
        sys.exit(1)

    print(f"\nResults saved to: {output_dir}")
    print("\nExpected performance:")
    print("  - BASE (baseline):      ~80% BA")
    print("  - BASE + OPRO:          ~86.9% BA")
    print("  - LoRA (baseline):      ~88% BA")
    print("  - LoRA + OPRO (BEST):   ~93.7% BA")
    print("=" * 80)


if __name__ == "__main__":
    main()
