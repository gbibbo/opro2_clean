# Auditoría de Scripts - Proyecto OPRO2

**Fecha:** 2025-12-27
**Total de scripts:** 12

---

## 📊 CLASIFICACIÓN DE SCRIPTS

### 1️⃣ SCRIPTS OFICIALES (Parte del flujo principal) - 6 scripts

| Script | Propósito | Usado en Jobs | Estado |
|--------|-----------|---------------|--------|
| `finetune_qwen_audio.py` | Fine-tuning LoRA del modelo Qwen2-Audio | `02_finetune.job` | ✅ Oficial |
| `evaluate_simple.py` | Evaluación de modelos en test set | `01_baseline.job`, `03_eval_lora.job`, etc. | ✅ Oficial |
| `opro_classic_optimize.py` | Optimización de prompts con OPRO (classic/open) | `04_opro_base.job`, `05_opro_lora.job` | ✅ Oficial |
| `opro_post_ft_v2.py` | Optimización de prompts post fine-tuning | Usado manualmente | ✅ Oficial |
| `run_complete_pipeline.py` | Pipeline completo (finetune + OPRO + eval) | `run_complete_pipeline.job` | ✅ Oficial |
| `statistical_analysis.py` | Funciones estadísticas core + CLI para 4 configs | `08_statistical_analysis.job` | ✅ Oficial |

**Funciones principales:**
- `statistical_analysis.py` contiene:
  - Cluster bootstrap para BA y deltas
  - Wilson score CIs para recalls
  - McNemar exact test
  - Holm-Bonferroni correction
  - Psychometric thresholds (DT50/75/90, SNR-75)
  - CLI limitado a 4 configs (baseline, base_opro, lora, lora_opro)

---

### 2️⃣ PARCHES TEMPORALES (Deben integrarse a oficiales) - 2 scripts

| Script | Propósito | Problema | Solución |
|--------|-----------|----------|----------|
| `compute_psychometric_for_all.py` | Calcula métricas psicométricas para 6 configs | Duplica funcionalidad | **Unificar en `compute_psychometric.py`** |
| `compute_psychometric_remaining.py` | Calcula métricas para 2 configs (varied) | Duplica funcionalidad | **Eliminar tras unificación** |

**Análisis:**
- Ambos scripts son casi idénticos
- Solo difieren en qué configuraciones procesan
- Deberían ser UN SOLO script que acepte argumentos para filtrar configs
- Propuesta: Crear `scripts/compute_psychometric.py` que reemplace a ambos

---

### 3️⃣ SCRIPTS BASURA (Temporales/Debug, eliminar) - 4 scripts

| Script | Propósito | Razón para eliminar |
|--------|-----------|---------------------|
| `create_consolidated_report.py` | Genera reporte de texto con métricas básicas | Solo para debugging, no usado en flujo oficial |
| `diagnose_base_nonspeech.py` | Diagnóstico de respuestas del modelo base en NonSpeech | Script de diagnóstico puntual, ya completado |
| `run_comprehensive_statistical_analysis.py` | Wrapper para ejecutar comparaciones pairwise | Nunca usado en jobs, funcionalidad duplicada |
| `test_statistical_analysis.py` | Testing rápido de funciones estadísticas | Script de prueba, no producción |

**Evidencia:**
```bash
# Ninguno de estos scripts está referenciado en jobs activos
$ grep -r "create_consolidated_report\|diagnose_base\|run_comprehensive\|test_statistical" slurm/*.job
# (vacío)
```

---

## ✅ VERIFICACIÓN

**Total de scripts:** 12
- Oficiales: 6
- Parches: 2
- Basura: 4
**Suma:** 6 + 2 + 4 = **12** ✅

---

## 🎯 PLAN DE ACCIÓN

### Paso 1: Unificar scripts psicométricos
- **Crear:** `scripts/compute_psychometric.py` (unifica `compute_psychometric_for_all.py` + `compute_psychometric_remaining.py`)
- **Eliminar:**
  - `compute_psychometric_for_all.py`
  - `compute_psychometric_remaining.py`
- **Actualizar:**
  - `slurm/psychometric_analysis.job` para usar el nuevo script

### Paso 2: Eliminar scripts basura
```bash
rm scripts/create_consolidated_report.py
rm scripts/diagnose_base_nonspeech.py
rm scripts/run_comprehensive_statistical_analysis.py
rm scripts/test_statistical_analysis.py
```

### Paso 3: Estado final
**Scripts oficiales finales:** 7
1. `finetune_qwen_audio.py`
2. `evaluate_simple.py`
3. `opro_classic_optimize.py`
4. `opro_post_ft_v2.py`
5. `run_complete_pipeline.py`
6. `statistical_analysis.py`
7. `compute_psychometric.py` ← **NUEVO (unificado)**

---

## 📝 NOTAS

- `statistical_analysis.py` tiene doble función:
  1. CLI para análisis de 4 configs específicas (usado en `08_statistical_analysis.job`)
  2. Biblioteca de funciones estadísticas (importado por otros scripts)

- El nuevo `compute_psychometric.py` debe:
  - Aceptar `--config` con lista de configs a procesar (o `--all` para todas)
  - Mantener compatibilidad con jobs existentes
  - Usar funciones de `statistical_analysis.py`
