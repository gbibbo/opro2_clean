#!/bin/bash
# =============================================================================
# Fix Failed Stages - Script Maestro
# =============================================================================
# Este script re-ejecuta SOLO los stages que fallaron en el pipeline anterior
# debido a errores de CUDA dentro del contenedor Apptainer.
#
# Stages a re-ejecutar:
#   - Stage 5: OPRO on LoRA model (falló con CUDA errors)
#   - Stage 7: Evaluate LoRA + OPRO (evaluó con modelo que tuvo CUDA errors)
#
# Stage 4 y 6 (BASE model) no se re-ejecutan porque usamos el modelo LoRA.
# =============================================================================

set -euo pipefail

REPO_CLEAN="/mnt/fast/nobackup/users/gb0048/opro2_clean"
cd "$REPO_CLEAN"

echo "================================================================================"
echo "FIX FAILED STAGES - Re-ejecución de Stages con errores de CUDA"
echo "================================================================================"
echo ""
echo "Problema detectado:"
echo "  - El job anterior (2025978) tuvo errores de CUDA dentro del contenedor"
echo "  - Stage 5 completó técnicamente pero todas las predicciones fallaron"
echo "  - Stage 7 evaluó pero obtuvo 0% accuracy debido a los errores"
echo ""
echo "Solución:"
echo "  - Re-ejecutar Stage 5 en un job individual con test de CUDA previo"
echo "  - Re-ejecutar Stage 7 después de que Stage 5 complete exitosamente"
echo ""
echo "================================================================================"
echo ""

# Enviar Stage 5
echo "[1/2] Enviando Stage 5: OPRO on LoRA model..."
JOB5=$(./slurm/tools/on_submit.sh sbatch "$REPO_CLEAN/slurm/fix_stage5_opro_lora.job" | grep -oP 'Submitted batch job \K\d+')

if [ -z "$JOB5" ]; then
    echo "ERROR: No se pudo enviar Stage 5"
    exit 1
fi

echo "  ✓ Stage 5 encolado: Job ID = $JOB5"
echo "  - Tiempo estimado: ~2.5 horas"
echo "  - Incluye test de CUDA antes de ejecutar"
echo ""

# Enviar Stage 7 con dependencia de Stage 5
echo "[2/2] Enviando Stage 7: Evaluate LoRA + OPRO (depende de Stage 5)..."
JOB7=$(./slurm/tools/on_submit.sh sbatch --dependency=afterok:$JOB5 "$REPO_CLEAN/slurm/fix_stage7_eval_lora_opro.job" | grep -oP 'Submitted batch job \K\d+')

if [ -z "$JOB7" ]; then
    echo "ERROR: No se pudo enviar Stage 7"
    exit 1
fi

echo "  ✓ Stage 7 encolado: Job ID = $JOB7"
echo "  - Tiempo estimado: ~2 horas"
echo "  - Comenzará automáticamente después de que Stage 5 complete"
echo ""

echo "================================================================================"
echo "JOBS ENVIADOS EXITOSAMENTE"
echo "================================================================================"
echo ""
echo "Pipeline de corrección:"
echo "  1. Stage 5 (Job $JOB5) → OPRO on LoRA model"
echo "  2. Stage 7 (Job $JOB7) → Evaluate LoRA + OPRO (espera a Stage 5)"
echo ""
echo "Tiempo total estimado: ~4.5 horas"
echo ""
echo "Monitoreo:"
echo "  - Ver cola:      ./slurm/tools/on_submit.sh squeue -u gb0048"
echo "  - Ver Stage 5:   tail -f logs/fix_stage5_${JOB5}.out"
echo "  - Ver Stage 7:   tail -f logs/fix_stage7_${JOB7}.out"
echo "  - Ver detalles:  ./slurm/tools/on_submit.sh scontrol show job $JOB5"
echo ""
echo "Resultados esperados:"
echo "  - results/complete_pipeline_seed42_opro_open/05_opro_lora/best_prompt.txt"
echo "  - results/complete_pipeline_seed42_opro_open/05_opro_lora/optimization_history.json"
echo "  - results/complete_pipeline_seed42_opro_open/07_eval_lora_opro/metrics.json"
echo "  - BA_clip esperado: ~93% (similar al experimento exitoso del 15 dic)"
echo ""
echo "================================================================================"
