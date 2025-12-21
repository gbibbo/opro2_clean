# OPRO2 - Optimización de Prompts para Detección de Habla

Pipeline completo de optimización de prompts (OPRO) para detección de habla con Qwen2-Audio y LoRA en Surrey HPC.

---

## Descripción

Este repositorio implementa un pipeline de 7 etapas para optimizar la detección de habla mediante:
- **OPRO (Optimization by PROmpting)**: Optimización automática de prompts usando un LLM local
- **LoRA (Low-Rank Adaptation)**: Fine-tuning eficiente del modelo Qwen2-Audio-7B-Instruct
- **Evaluación psicoacústica**: Medición de rendimiento bajo 22 condiciones independientes

---

## Últimos Resultados Experimentales

### Experimento 1: Pipeline Completo Seed 42 (OPRO Clásico) ✅

**Configuración:**
- Seed: 42
- Modelo: Qwen2-Audio-7B-Instruct + LoRA
- Checkpoint: `checkpoints/qwen_lora_seed42/final`
- OPRO: 15 iteraciones, 20 muestras por iteración, 8 candidatos
- Fecha: 15 de diciembre 2025

**Mejor Prompt Encontrado:**
```
Decide the dominant content.
Definitions:
- SPEECH = human voice, spoken words, syllables, conversational cues.
- NONSPEECH = music, tones/beeps, environmental noise, silence.
Output exactly: SPEECH or NONSPEECH.
```

**Métricas Finales (Test Set - 21,340 muestras):**
- **BA_clip:** 93.02%
- **BA_conditions:** 93.48%
- **Speech Accuracy:** 98.35%
- **Nonspeech Accuracy:** 87.69%

**Desglose por Dimensión:**
| Dimensión | BA | Condiciones |
|-----------|-----|------------|
| Duration | 89.14% | 8 condiciones (20ms-1000ms) |
| SNR | 97.11% | 6 condiciones (-10dB a 20dB) |
| Reverb | 93.71% | 4 condiciones (none, 0.3s, 1.0s, 2.5s) |
| Filter | 93.94% | 4 condiciones (none, bandpass, lowpass, highpass) |

**Rendimiento por Condición (Top 5):**
1. SNR 5dB: 97.32% BA
2. SNR 10dB: 97.22% BA
3. SNR -10dB/20dB/-5dB/0dB: ~97.01% BA
4. Filter Bandpass: 94.33% BA
5. Filter Lowpass: 94.12% BA

**Rendimiento por Condición (Bottom 5):**
1. Duration 20ms: 80.93% BA
2. Duration 40ms: 84.85% BA
3. Duration 60ms: 87.32% BA
4. Duration 80ms: 88.04% BA
5. Duration 100ms: 90.82% BA

**Evolución del OPRO:**
- Iteración 1: 90% accuracy (prompt inicial)
- Mejor iteración: Iter 76 con 100% accuracy en muestra de validación
- Total de prompts generados: 121 (15 iteraciones × 8 candidatos)
- Top prompts recurrentes:
  - "Does this audio contain human speech? Answer exactly one token: SPEECH or NONSPEECH." → 95%
  - "Decide the dominant content..." → 100% (mejor)
  - "Label SPEECH only if human voice is clearly present..." → 95%

---

### Experimento 2: Pipeline con OPRO Open Prompts ❌

**Configuración:**
- Seed: 42
- Modelo: Qwen2-Audio-7B-Instruct + LoRA
- Checkpoint: `checkpoints/qwen_lora_seed42/final`
- OPRO: 15 iteraciones, 20 muestras por iteración, 8 candidatos
- Modo: Open prompts (sin restricciones de formato)
- Fecha: 20 de diciembre 2025

**Mejor Prompt Encontrado:**
```
Does this audio contain human speech? Answer exactly one token: SPEECH or NONSPEECH.
```

**Métricas Finales (Test Set - 21,340 muestras):**
- **BA_clip:** 0.00% ❌
- **BA_conditions:** 0.00% ❌
- **Speech Accuracy:** 0.00% ❌
- **Nonspeech Accuracy:** 0.00% ❌

**Análisis del Fracaso:**

El experimento con prompts abiertos falló completamente. Análisis de las 121 evaluaciones:
- **Todas las iteraciones obtuvieron 0% accuracy**
- El modelo no logró generar respuestas válidas ("SPEECH" o "NONSPEECH")
- Los prompts generados fueron variados pero ninguno funcionó:
  - Prompts con ejemplos
  - Prompts con definiciones técnicas (formants, syllabic rhythm)
  - Prompts con diferentes formatos de instrucción
  - Prompts con XML-style tags

**Prompts generados (muestra):**
1. "Does this audio contain human speech? Answer exactly one token: SPEECH or NONSPEECH." → 0%
2. "Binary decision. Output exactly one token: SPEECH or NONSPEECH." → 0%
3. "Decide the dominant content.\nDefinitions:\n- SPEECH = human voice, spoken words, syllables, conversational cues.\n- NONSPEECH = music, tones/beeps, environmental noise, silence.\nOutput exactly: SPEECH or NONSPEECH." → 0%
4. "Focus on cues of human vocal tract (formants, syllabic rhythm, consonant onsets).\nAnswer exactly: SPEECH or NONSPEECH." → 0%
5. "You will answer with one token only.\n<question>Does this audio contain human speech?</question>\n<answer>SPEECH or NONSPEECH only</answer>" → 0%

**Hipótesis sobre el fracaso:**
1. **Problema de normalización:** Los prompts open pueden haber generado respuestas que el sistema de normalización no procesó correctamente
2. **Problema de formato:** El modelo puede haber generado respuestas en un formato no esperado por el evaluador
3. **Problema de configuración:** Puede haber un bug en el script `opro_open_prompts.py` que no se detectó
4. **Incompatibilidad con el checkpoint:** El checkpoint LoRA puede requerir un formato específico de prompt que los prompts abiertos no respetan

**Recomendaciones:**
- ✅ Usar OPRO clásico (con restricciones de formato) que demostró excelentes resultados (93% BA)
- ❌ No usar OPRO open prompts hasta investigar y corregir el problema
- 🔍 Investigar el sistema de normalización en `src/qsm/utils/normalize.py`
- 🔍 Revisar el script `scripts/opro_open_prompts.py` para detectar posibles bugs

---

## Comparativa de Experimentos

| Experimento | BA_clip | Speech Acc | Nonspeech Acc | Status |
|-------------|---------|------------|---------------|--------|
| OPRO Clásico (seed 42) | **93.02%** | **98.35%** | **87.69%** | ✅ Exitoso |
| OPRO Open Prompts (seed 42) | 0.00% | 0.00% | 0.00% | ❌ Fallido |
| Diferencia | -93.02% | -98.35% | -87.69% | - |

**Conclusión:** El enfoque clásico de OPRO con restricciones de formato es significativamente superior al enfoque de prompts abiertos, que falló completamente.

---

## Pipeline de 7 Etapas

1. **Evaluación psicoacústica (baseline)** - Establece la línea base
2. **LoRA finetuning** - Entrena adaptadores sobre el modelo base
3. **Evaluación base vs LoRA** - Compara ambos modelos
4. **OPRO en modelo base** - Optimiza prompts para el modelo base
5. **OPRO en modelo LoRA** - Re-optimiza prompts para el modelo fine-tuned
6. **Evaluación base + OPRO** - Eval con prompts optimizados
7. **Evaluación LoRA + OPRO** - Eval final con LoRA + OPRO ⭐

---

## Estructura del Proyecto

```
opro2_clean/
├── README.md                          # Este archivo
├── MANIFEST.md                        # Inventario completo de archivos
├── CLAUDE.md                          # Instrucciones para Claude Code
├── requirements.txt                   # Dependencias Python
├── config.yaml                        # Configuración global
│
├── scripts/
│   ├── evaluate_simple.py             # Evaluación principal (Etapas 1,3,6,7)
│   ├── finetune_qwen_audio.py         # LoRA training (Etapa 2)
│   ├── opro_classic_optimize.py       # OPRO clásico (Etapa 4,5) ✅
│   ├── opro_post_ft_v2.py             # OPRO post-FT
│   ├── opro_open_prompts.py           # OPRO open prompts ❌
│   └── run_complete_pipeline.py       # Wrapper completo
│
├── slurm/                             # Jobs de SLURM
│   ├── tools/on_submit.sh             # Wrapper para ejecutar comandos SLURM
│   └── *.job                          # Scripts de jobs
│
├── src/qsm/                           # Código fuente
│   ├── models/qwen_audio.py           # Wrapper del modelo
│   └── utils/normalize.py             # Utilidades de normalización
│
├── checkpoints/                       # Checkpoints LoRA
│   └── qwen_lora_seed42/
│       └── final/                     # Checkpoint final usado en experimentos
│
├── results/                           # Resultados de evaluación
│   ├── complete_pipeline_seed42/      # Experimento 1 ✅
│   │   ├── 05_opro_lora/
│   │   │   ├── optimization_history.json
│   │   │   └── best_prompt.txt
│   │   └── 03_eval_lora/
│   │       └── metrics.json
│   │
│   └── complete_pipeline_seed42_opro_open/  # Experimento 2 ❌
│       ├── 05_opro_lora/
│       │   ├── optimization_history.json
│       │   └── best_prompt.txt
│       └── 07_eval_lora_opro/
│           └── metrics.json
│
└── logs/                              # Logs de SLURM
```

---

## Requisitos

### Sistema
- Python >= 3.10
- CUDA >= 11.8
- **GPU:** 40GB+ VRAM para training, 24GB para inference (RTX 3090, A6000, V100)
- RAM: 48-64GB
- Disco: 100GB+ libres

### Instalación

```bash
# Clonar repositorio
cd /mnt/fast/nobackup/users/gb0048/opro2_clean

# Instalar dependencias
pip install -r requirements.txt

# Verificar GPU
nvidia-smi
```

**Dependencias principales:**
- `torch>=2.0.0`, `torchaudio>=2.0.0`
- `transformers>=4.40.0`
- `peft>=0.10.0` (LoRA)
- `pandas>=2.0.0`, `pyarrow>=15.0.0`
- `librosa>=0.10.1`, `soundfile>=0.12.1`

---

## Uso en Surrey HPC

### Ejecutar Pipeline Completo

```bash
# Vía wrapper de submit (recomendado)
./slurm/tools/on_submit.sh sbatch slurm/00_run_complete_pipeline.job

# Ver cola de jobs
./slurm/tools/on_submit.sh squeue -u gb0048

# Ver detalles de un job
./slurm/tools/on_submit.sh scontrol show job JOBID

# Ver histórico
./slurm/tools/on_submit.sh sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,ReqMem,MaxRSS
```

### Ejecutar Etapas Individuales

```bash
# Etapa 2: LoRA Training
./slurm/tools/on_submit.sh sbatch slurm/01_finetune_lora.job 42

# Etapa 5: OPRO en LoRA (clásico - recomendado)
./slurm/tools/on_submit.sh sbatch slurm/03_opro_lora.job 42

# Etapa 7: Evaluación final
./slurm/tools/on_submit.sh sbatch slurm/07_eval_lora_opro.job 42
```

---

## Configuración Técnica

### LoRA

```yaml
lora:
  r: 64                     # Rank
  alpha: 16                 # Scaling
  dropout: 0.05
  task_type: CAUSAL_LM
  target_modules:
    - q_proj, k_proj, v_proj, o_proj
    - gate_proj, up_proj, down_proj
```

### Entrenamiento

- **Quantization:** 4-bit (QLoRA)
- **Batch size:** 2 × 4 gradient accumulation = 8 effective
- **Learning rate:** 5e-5
- **Epochs:** 3
- **Gradient checkpointing:** Enabled

### OPRO Clásico (Recomendado)

- **Optimizer LLM:** Qwen/Qwen2.5-7B-Instruct (local)
- **Iterations:** 15
- **Samples per iteration:** 20
- **Candidates per iteration:** 8
- **Top-k memory:** 10 mejores prompts
- **Reward function:** Balanced Accuracy

---

## Archivos de Resultados

### Experimento 1 (Exitoso)

- **Historia de optimización:** [results/complete_pipeline_seed42/05_opro_lora/optimization_history.json](results/complete_pipeline_seed42/05_opro_lora/optimization_history.json)
- **Mejor prompt:** [results/complete_pipeline_seed42/05_opro_lora/best_prompt.txt](results/complete_pipeline_seed42/05_opro_lora/best_prompt.txt)
- **Métricas finales:** [results/complete_pipeline_seed42/03_eval_lora/metrics.json](results/complete_pipeline_seed42/03_eval_lora/metrics.json)

### Experimento 2 (Fallido)

- **Historia de optimización:** [results/complete_pipeline_seed42_opro_open/05_opro_lora/optimization_history.json](results/complete_pipeline_seed42_opro_open/05_opro_lora/optimization_history.json)
- **Mejor prompt:** [results/complete_pipeline_seed42_opro_open/05_opro_lora/best_prompt.txt](results/complete_pipeline_seed42_opro_open/05_opro_lora/best_prompt.txt)
- **Métricas finales:** [results/complete_pipeline_seed42_opro_open/07_eval_lora_opro/metrics.json](results/complete_pipeline_seed42_opro_open/07_eval_lora_opro/metrics.json)

---

## Troubleshooting

### Error: "CUDA out of memory"
```bash
# Reducir batch size
python scripts/evaluate_simple.py --batch_size 20  # default: 50

# Configurar memoria expandible
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

### Error: "Checkpoint not found"
```bash
# Verificar que existe el checkpoint
ls -la checkpoints/qwen_lora_seed42/final/

# Si no existe, entrenar primero
./slurm/tools/on_submit.sh sbatch slurm/01_finetune_lora.job 42
```

### Jobs con DependencyNeverSatisfied
```bash
# Ver detalles del job
./slurm/tools/on_submit.sh scontrol show job JOBID | sed -n '1,120p'

# Cancelar y reenviar sin dependencia
./slurm/tools/on_submit.sh scancel JOBID
./slurm/tools/on_submit.sh sbatch slurm/script.job
```

---

## Documentación Adicional

- **[CLAUDE.md](CLAUDE.md):** Reglas operativas para Claude Code en Surrey HPC
- **[MANIFEST.md](MANIFEST.md):** Inventario completo de archivos
- **[config.yaml](config.yaml):** Configuración global del proyecto
- **[RUN_PIPELINE.md](RUN_PIPELINE.md):** Guía detallada de ejecución

---

## Próximos Pasos

### Investigaciones Pendientes

1. **Debuggear OPRO Open Prompts:**
   - Revisar sistema de normalización en `src/qsm/utils/normalize.py`
   - Verificar compatibilidad con checkpoint LoRA
   - Añadir logging detallado para entender por qué falla

2. **Optimizaciones Posibles:**
   - Probar con diferentes seeds (43, 44, 45)
   - Experimentar con diferentes configuraciones de LoRA (r=32, r=128)
   - Probar otros LLMs para OPRO (Llama, Mistral)

3. **Análisis de Errores:**
   - Investigar por qué duration corta tiene peor rendimiento
   - Analizar muestras mal clasificadas en nonspeech
   - Estudiar confusiones específicas por condición

---

## Contacto

**Proyecto:** OPRO2 - Optimización de Prompts para Detección de Habla
**Ubicación:** Surrey HPC (aisurrey-submit01.surrey.ac.uk)
**Working Directory:** `/mnt/fast/nobackup/users/gb0048/opro2_clean`

Para preguntas o problemas:
1. Revisar logs en `logs/`
2. Consultar `CLAUDE.md` para comandos SLURM
3. Verificar estado de jobs con `./slurm/tools/on_submit.sh squeue -u gb0048`

---

**Última actualización:** 21 de diciembre 2025
**Versión:** 2.0
**Status:** 🟢 OPRO Clásico funcional | 🔴 OPRO Open Prompts requiere debugging
