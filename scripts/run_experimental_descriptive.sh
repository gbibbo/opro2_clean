#!/bin/bash
# =============================================================================
# EXPERIMENTAL: OPRO with Descriptive Open Prompts
# =============================================================================
# Tests if open-ended prompts like "What do you hear?" work better than
# constrained prompts for speech detection.
#
# Key differences from standard OPRO:
#   - Uses descriptive seed prompts (e.g., "What do you hear in this audio?")
#   - Allows free-form answers (e.g., "I hear voices in a rural environment")
#   - Uses 30 samples per iteration (vs 20 in standard OPRO)
#   - Relies on synonym-based normalization to classify responses
# =============================================================================

set -euo pipefail

REPO_CLEAN="/mnt/fast/nobackup/users/gb0048/opro2_clean"
cd "$REPO_CLEAN"

echo "================================================================================"
echo "EXPERIMENTAL: OPRO with Descriptive Open Prompts"
echo "================================================================================"
echo ""
echo "This experiment tests whether open-ended descriptive prompts can outperform"
echo "traditional constrained prompts for speech detection tasks."
echo ""
echo "Approach:"
echo "  - Seed prompts encourage free-form descriptions"
echo "  - Model can respond naturally (e.g., 'I hear background voices')"
echo "  - Normalization system classifies based on content words (voice, music, etc.)"
echo ""
echo "Configuration:"
echo "  - Model: BASE (without LoRA)"
echo "  - Samples per iteration: 30 (increased for better gradient signal)"
echo "  - Decoding mode: open (unrestricted generation)"
echo "  - Seed prompts: prompts/open_descriptive_seeds.json"
echo ""
echo "================================================================================"
echo ""

# Enviar OPRO optimization
echo "[1/2] Enviando OPRO optimization con prompts descriptivos..."
JOB_OPRO=$(./slurm/tools/on_submit.sh sbatch "$REPO_CLEAN/slurm/experimental_opro_descriptive.job" | grep -oP 'Submitted batch job \K\d+')

if [ -z "$JOB_OPRO" ]; then
    echo "ERROR: No se pudo enviar job de OPRO"
    exit 1
fi

echo "  ✓ OPRO job encolado: Job ID = $JOB_OPRO"
echo "  - Tiempo estimado: ~3 horas"
echo "  - Incluye test de CUDA antes de ejecutar"
echo ""

# Enviar evaluación con dependencia
echo "[2/2] Enviando evaluación en test set (depende de OPRO)..."
JOB_EVAL=$(./slurm/tools/on_submit.sh sbatch --dependency=afterok:$JOB_OPRO "$REPO_CLEAN/slurm/experimental_eval_descriptive.job" | grep -oP 'Submitted batch job \K\d+')

if [ -z "$JOB_EVAL" ]; then
    echo "ERROR: No se pudo enviar job de evaluación"
    exit 1
fi

echo "  ✓ Eval job encolado: Job ID = $JOB_EVAL"
echo "  - Tiempo estimado: ~2 horas"
echo "  - Comenzará automáticamente después de OPRO"
echo ""

echo "================================================================================"
echo "JOBS ENVIADOS EXITOSAMENTE"
echo "================================================================================"
echo ""
echo "Pipeline experimental:"
echo "  1. OPRO Optimization (Job $JOB_OPRO) → Buscar mejor prompt descriptivo"
echo "  2. Test Set Evaluation (Job $JOB_EVAL) → Medir rendimiento final"
echo ""
echo "Tiempo total estimado: ~5 horas"
echo ""
echo "Monitoreo:"
echo "  - Ver cola:        ./slurm/tools/on_submit.sh squeue -u gb0048"
echo "  - Ver OPRO log:    tail -f logs/opro_descriptive_${JOB_OPRO}.out"
echo "  - Ver eval log:    tail -f logs/eval_descriptive_${JOB_EVAL}.out"
echo "  - Ver detalles:    ./slurm/tools/on_submit.sh scontrol show job $JOB_OPRO"
echo ""
echo "Resultados esperados:"
echo "  - results/experimental_opro_descriptive_seed42/base/best_prompt.txt"
echo "  - results/experimental_opro_descriptive_seed42/base/optimization_history.json"
echo "  - results/experimental_opro_descriptive_seed42/eval_base/metrics.json"
echo ""
echo "Hipótesis:"
echo "  Si este experimento funciona bien, demuestra que prompts descriptivos"
echo "  pueden ser más efectivos que prompts binarios restrictivos."
echo ""
echo "  Ejemplo de respuesta esperada:"
echo "    Prompt: 'What do you hear in this audio?'"
echo "    Modelo: 'I hear voices talking in the background'"
echo "    Sistema: Clasifica como SPEECH (por la palabra 'voices')"
echo ""
echo "================================================================================"
