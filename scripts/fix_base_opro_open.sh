#!/bin/bash
# =============================================================================
# Fix BASE + OPRO Open - Script Maestro
# =============================================================================
# Este script ejecuta los stages faltantes:
#   - Stage 4: OPRO Open on BASE model
#   - Stage 6: Evaluate BASE + OPRO Open
# =============================================================================

set -euo pipefail

REPO_CLEAN="/mnt/fast/nobackup/users/gb0048/opro2_clean"
cd "$REPO_CLEAN"

echo "================================================================================"
echo "FIX BASE + OPRO OPEN - Experimento faltante"
echo "================================================================================"
echo ""
echo "Experimentos a ejecutar:"
echo "  - Stage 4: OPRO Open en modelo BASE (sin LoRA)"
echo "  - Stage 6: Evaluación BASE + OPRO Open en test set"
echo ""
echo "================================================================================"
echo ""

# Enviar Stage 4
echo "[1/2] Enviando Stage 4: OPRO Open on BASE model..."
JOB4=$(./slurm/tools/on_submit.sh sbatch "$REPO_CLEAN/slurm/fix_stage4_opro_base.job" | grep -oP 'Submitted batch job \K\d+')

if [ -z "$JOB4" ]; then
    echo "ERROR: No se pudo enviar Stage 4"
    exit 1
fi

echo "  ✓ Stage 4 encolado: Job ID = $JOB4"
echo "  - Tiempo estimado: ~2.5 horas"
echo "  - Incluye test de CUDA antes de ejecutar"
echo ""

# Enviar Stage 6 con dependencia de Stage 4
echo "[2/2] Enviando Stage 6: Evaluate BASE + OPRO Open (depende de Stage 4)..."
JOB6=$(./slurm/tools/on_submit.sh sbatch --dependency=afterok:$JOB4 "$REPO_CLEAN/slurm/fix_stage6_eval_base_opro.job" | grep -oP 'Submitted batch job \K\d+')

if [ -z "$JOB6" ]; then
    echo "ERROR: No se pudo enviar Stage 6"
    exit 1
fi

echo "  ✓ Stage 6 encolado: Job ID = $JOB6"
echo "  - Tiempo estimado: ~2 horas"
echo "  - Comenzará automáticamente después de que Stage 4 complete"
echo ""

echo "================================================================================"
echo "JOBS ENVIADOS EXITOSAMENTE"
echo "================================================================================"
echo ""
echo "Pipeline de corrección:"
echo "  1. Stage 4 (Job $JOB4) → OPRO Open on BASE model"
echo "  2. Stage 6 (Job $JOB6) → Evaluate BASE + OPRO Open (espera a Stage 4)"
echo ""
echo "Tiempo total estimado: ~4.5 horas"
echo ""
echo "Monitoreo:"
echo "  - Ver cola:      ./slurm/tools/on_submit.sh squeue -u gb0048"
echo "  - Ver Stage 4:   tail -f logs/fix_stage4_${JOB4}.out"
echo "  - Ver Stage 6:   tail -f logs/fix_stage6_${JOB6}.out"
echo "  - Ver detalles:  ./slurm/tools/on_submit.sh scontrol show job $JOB4"
echo ""
echo "Resultados esperados:"
echo "  - results/complete_pipeline_seed42_opro_open/04_opro_base/best_prompt.txt"
echo "  - results/complete_pipeline_seed42_opro_open/04_opro_base/optimization_history.json"
echo "  - results/complete_pipeline_seed42_opro_open/06_eval_base_opro/metrics.json"
echo "  - BA_clip esperado: ~88% (similar al OPRO Clásico en BASE)"
echo ""
echo "================================================================================"
