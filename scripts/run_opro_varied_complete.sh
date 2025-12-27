#!/bin/bash
# =============================================================================
# OPRO with Varied Seed Prompts - Complete Pipeline
# =============================================================================
# Executes OPRO with 15 varied seed prompts on BOTH:
#   - BASE model (no LoRA)
#   - LoRA model (fine-tuned)
# Then evaluates both on test set
# =============================================================================

set -euo pipefail

REPO_CLEAN="/mnt/fast/nobackup/users/gb0048/opro2_clean"
cd "$REPO_CLEAN"

echo "================================================================================"
echo "OPRO with Varied Seed Prompts - Complete Pipeline"
echo "================================================================================"
echo ""
echo "This experiment runs OPRO with 15 diverse seed prompts (descriptive, binary,"
echo "with definitions, with examples, multiple choice, etc.) on BOTH models:"
echo "  - BASE model (without LoRA)"
echo "  - LoRA model (fine-tuned)"
echo ""
echo "Then evaluates both optimized prompts on the test set."
echo ""
echo "================================================================================"
echo ""

# Submit OPRO on BASE
echo "[1/4] Submitting OPRO optimization on BASE model..."
JOB_OPRO_BASE=$(./slurm/tools/on_submit.sh sbatch "$REPO_CLEAN/slurm/opro_varied_base.job" | grep -oP 'Submitted batch job \K\d+')

if [ -z "$JOB_OPRO_BASE" ]; then
    echo "ERROR: Could not submit OPRO BASE job"
    exit 1
fi

echo "  ✓ OPRO BASE enqueued: Job ID = $JOB_OPRO_BASE"
echo "  - Estimated time: ~3 hours"
echo ""

# Submit OPRO on LoRA
echo "[2/4] Submitting OPRO optimization on LoRA model..."
JOB_OPRO_LORA=$(./slurm/tools/on_submit.sh sbatch "$REPO_CLEAN/slurm/opro_varied_lora.job" | grep -oP 'Submitted batch job \K\d+')

if [ -z "$JOB_OPRO_LORA" ]; then
    echo "ERROR: Could not submit OPRO LoRA job"
    exit 1
fi

echo "  ✓ OPRO LoRA enqueued: Job ID = $JOB_OPRO_LORA"
echo "  - Estimated time: ~3 hours"
echo ""

# Submit eval for BASE (depends on OPRO BASE)
echo "[3/4] Submitting evaluation for BASE + OPRO varied..."
JOB_EVAL_BASE=$(./slurm/tools/on_submit.sh sbatch --dependency=afterok:$JOB_OPRO_BASE "$REPO_CLEAN/slurm/eval_varied_base.job" | grep -oP 'Submitted batch job \K\d+')

if [ -z "$JOB_EVAL_BASE" ]; then
    echo "ERROR: Could not submit eval BASE job"
    exit 1
fi

echo "  ✓ Eval BASE enqueued: Job ID = $JOB_EVAL_BASE"
echo "  - Will start after OPRO BASE completes"
echo "  - Estimated time: ~2 hours"
echo ""

# Submit eval for LoRA (depends on OPRO LoRA)
echo "[4/4] Submitting evaluation for LoRA + OPRO varied..."
JOB_EVAL_LORA=$(./slurm/tools/on_submit.sh sbatch --dependency=afterok:$JOB_OPRO_LORA "$REPO_CLEAN/slurm/eval_varied_lora.job" | grep -oP 'Submitted batch job \K\d+')

if [ -z "$JOB_EVAL_LORA" ]; then
    echo "ERROR: Could not submit eval LoRA job"
    exit 1
fi

echo "  ✓ Eval LoRA enqueued: Job ID = $JOB_EVAL_LORA"
echo "  - Will start after OPRO LoRA completes"
echo "  - Estimated time: ~2 hours"
echo ""

echo "================================================================================"
echo "ALL JOBS SUBMITTED SUCCESSFULLY"
echo "================================================================================"
echo ""
echo "Pipeline:"
echo "  1. OPRO BASE    (Job $JOB_OPRO_BASE) → Optimize prompts on BASE model"
echo "  2. OPRO LoRA    (Job $JOB_OPRO_LORA) → Optimize prompts on LoRA model"
echo "  3. Eval BASE    (Job $JOB_EVAL_BASE) → Evaluate BASE + best prompt"
echo "  4. Eval LoRA    (Job $JOB_EVAL_LORA) → Evaluate LoRA + best prompt"
echo ""
echo "Total estimated time: ~5 hours (OPRO and eval can run in parallel)"
echo ""
echo "Monitoring:"
echo "  - Queue:           ./slurm/tools/on_submit.sh squeue -u gb0048"
echo "  - OPRO BASE log:   tail -f logs/opro_varied_base_${JOB_OPRO_BASE}.out"
echo "  - OPRO LoRA log:   tail -f logs/opro_varied_lora_${JOB_OPRO_LORA}.out"
echo "  - Eval BASE log:   tail -f logs/eval_varied_base_${JOB_EVAL_BASE}.out"
echo "  - Eval LoRA log:   tail -f logs/eval_varied_lora_${JOB_EVAL_LORA}.out"
echo ""
echo "Results will be saved to:"
echo "  - results/opro_varied_seed42/base/best_prompt.txt"
echo "  - results/opro_varied_seed42/lora/best_prompt.txt"
echo "  - results/opro_varied_seed42/eval_base/metrics.json"
echo "  - results/opro_varied_seed42/eval_lora/metrics.json"
echo ""
echo "================================================================================"
