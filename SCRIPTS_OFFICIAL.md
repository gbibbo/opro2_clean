# Scripts Oficiales - Proyecto OPRO2

**Última actualización:** 2025-12-27
**Total de scripts oficiales:** 7

---

## 📋 SCRIPTS OFICIALES DEL PROYECTO

### 1. Pipeline Principal

#### `run_complete_pipeline.py`
**Propósito:** Ejecuta el pipeline completo de experimentación
- Fine-tuning con LoRA
- Optimización de prompts con OPRO
- Evaluación de todos los modelos

**Usado en:**
- `slurm/run_complete_pipeline.job`

---

### 2. Fine-tuning

#### `finetune_qwen_audio.py`
**Propósito:** Fine-tuning de Qwen2-Audio con LoRA
- Configuración LoRA (rank=128, alpha=256)
- Training con gradient accumulation
- Checkpoint management

**Usado en:**
- `slurm/02_finetune.job`
- Llamado por `run_complete_pipeline.py`

---

### 3. Optimización de Prompts (OPRO)

#### `opro_classic_optimize.py`
**Propósito:** Optimización de prompts usando meta-LLM (Qwen2.5-7B)
- Soporta dos modos:
  - **Classic:** Instrucciones específicas para SPEECH/NON-SPEECH
  - **Open:** Instrucciones abiertas para formatos diversos
- Iterative refinement con feedback
- Evaluación en validation set

**Usado en:**
- `slurm/04_opro_base.job` (base model)
- `slurm/05_opro_lora.job` (LoRA model)
- Llamado por `run_complete_pipeline.py`

**Parámetros clave:**
- `--mode`: `classic` o `open`
- `--meta_model`: Modelo para generar prompts
- `--num_iterations`: Número de iteraciones OPRO

#### `opro_post_ft_v2.py`
**Propósito:** OPRO después del fine-tuning
- Optimización post-LoRA
- Usado manualmente para experimentos adicionales

---

### 4. Evaluación

#### `evaluate_simple.py`
**Propósito:** Evaluación de modelos en test set
- Carga modelo base o LoRA
- Procesa test metadata (21,340 samples)
- Genera predictions.csv y metrics.json
- Calcula Balanced Accuracy, F1, Precision, Recall

**Usado en:**
- `slurm/01_baseline.job`
- `slurm/03_eval_lora.job`
- `slurm/06_eval_base_opro.job`
- `slurm/07_eval_lora_opro.job`
- Llamado por `run_complete_pipeline.py`

**Output:**
- `predictions.csv`: clip_id, true_label, pred_label, response
- `metrics.json`: BA, F1, precision, recall

---

### 5. Análisis Estadístico

#### `statistical_analysis.py`
**Propósito:** Funciones estadísticas core + CLI
- **Funciones de biblioteca:**
  - `cluster_bootstrap_ba()`: Bootstrap a nivel de clip para BA
  - `cluster_bootstrap_delta_ba()`: Bootstrap para diferencias
  - `cluster_bootstrap_thresholds()`: Bootstrap para DT/SNR thresholds
  - `compute_recalls_with_wilson()`: Wilson score CIs para recalls
  - `mcnemar_exact_test()`: Test exacto de McNemar
  - `holm_bonferroni_correction()`: Corrección de múltiples comparaciones

- **CLI (limitado a 4 configs):**
  - Comparaciones pairwise: baseline vs base_opro, lora vs lora_opro
  - Genera `statistical_analysis.json` y `statistical_report.txt`

**Usado en:**
- `slurm/08_statistical_analysis.job` (CLI mode)
- Importado por `compute_psychometric.py` (library mode)

**Parámetros CLI:**
```bash
python statistical_analysis.py \
  --baseline <path> \
  --base_opro <path> \
  --lora <path> \
  --lora_opro <path> \
  --output_dir <dir> \
  --n_bootstrap 10000 \
  --seed 42
```

#### `compute_psychometric.py`
**Propósito:** Cálculo de thresholds psicométricos para cualquier configuración
- DT50, DT75, DT90 (Duration Thresholds)
- SNR-75 (SNR threshold at 75% accuracy)
- Cluster bootstrap con CIs
- Soporta procesamiento selectivo de configs

**Usado en:**
- `slurm/psychometric_analysis.job` (todas las configs)
- `slurm/psychometric_remaining.job` (configs específicas)

**Configuraciones disponibles:**
1. `baseline` - Baseline (Hand-crafted)
2. `lora_hand` - LoRA + Hand-crafted
3. `base_opro_classic` - Base + OPRO (Classic)
4. `lora_opro_classic` - LoRA + OPRO (Classic)
5. `base_opro_open` - Base + OPRO (Open)
6. `lora_opro_open` - LoRA + OPRO (Open)
7. `base_opro_varied` - Base + OPRO (Varied seeds)
8. `lora_opro_varied` - LoRA + OPRO (Varied seeds)

**Ejemplos de uso:**
```bash
# Procesar todas las configuraciones
python compute_psychometric.py --all

# Procesar configuraciones específicas
python compute_psychometric.py --configs baseline lora_hand

# Procesar solo varied seeds
python compute_psychometric.py --configs base_opro_varied lora_opro_varied
```

**Output:**
- `<config>_psychometric.json`: Resultados individuales
- `all_psychometric_thresholds.json`: Resultados combinados
- `psychometric_report.txt`: Reporte legible

---

## 🔄 Flujo de Trabajo

### Experimento Completo
```
1. run_complete_pipeline.py
   ├─> finetune_qwen_audio.py (stage 2)
   ├─> opro_classic_optimize.py (stages 4-5)
   └─> evaluate_simple.py (stages 1,3,6,7)

2. compute_psychometric.py --all
   └─> Usa funciones de statistical_analysis.py

3. statistical_analysis.py (CLI)
   └─> Comparaciones pairwise
```

### Análisis Psicométrico Independiente
```
# Todas las configuraciones
sbatch slurm/psychometric_analysis.job

# Configuraciones específicas
sbatch slurm/psychometric_remaining.job
```

---

## 📊 Dependencias entre Scripts

```
run_complete_pipeline.py
  ├─ imports: evaluate_simple, opro_classic_optimize (indirectamente)
  └─ calls: finetune_qwen_audio, evaluate_simple

compute_psychometric.py
  └─ imports: statistical_analysis (funciones)

opro_classic_optimize.py
  └─ standalone (usa Qwen2-Audio y Qwen2.5-7B)

statistical_analysis.py
  └─ standalone + biblioteca

evaluate_simple.py
  └─ standalone

finetune_qwen_audio.py
  └─ standalone
```

---

## ✅ Verificación Final

**Antes de la limpieza:** 12 scripts
- 6 oficiales originales
- 2 parches temporales
- 4 scripts basura

**Después de la limpieza:** 7 scripts oficiales
- 6 scripts originales mantenidos
- 1 script nuevo unificado (`compute_psychometric.py`)
- 0 scripts temporales/basura

**Scripts eliminados:**
- ❌ `compute_psychometric_for_all.py` (reemplazado por `compute_psychometric.py`)
- ❌ `compute_psychometric_remaining.py` (reemplazado por `compute_psychometric.py`)
- ❌ `create_consolidated_report.py` (debugging temporal)
- ❌ `diagnose_base_nonspeech.py` (diagnóstico puntual)
- ❌ `run_comprehensive_statistical_analysis.py` (nunca usado)
- ❌ `test_statistical_analysis.py` (testing temporal)

---

## 📝 Mantenimiento

Para verificar que no hay scripts huérfanos:
```bash
# Contar scripts
ls -1 scripts/*.py | wc -l  # Debe ser 7

# Listar scripts
ls -1 scripts/*.py

# Verificar uso en jobs
for script in scripts/*.py; do
    echo "=== $(basename $script) ==="
    grep -l "$(basename $script)" slurm/*.job 2>/dev/null || echo "  No usado en jobs"
done
```
