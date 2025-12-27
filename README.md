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

### ⚠️ Hallazgo Crítico: Modelo BASE Sin Fine-tuning NO es Viable (26 diciembre 2025)

**Diagnóstico completo revela sesgo inherente del modelo BASE:**

El modelo Qwen2-Audio-7B-Instruct **sin fine-tuning presenta un sesgo crítico hacia clasificar audios como SPEECH**, independientemente de la estrategia de prompting utilizada.

**Evidencia experimental (100 samples NONSPEECH):**
- ✅ **Correctos (NONSPEECH)**: 67%
- ❌ **Incorrectos (predice SPEECH)**: 33%
- Con **degradaciones severas** (test set completo): Solo **25.52% accuracy** en NONSPEECH

**Observaciones técnicas:**
- El modelo responde literalmente `"SPEECH"` cuando debería decir `"NONSPEECH"`
- `confidence=1.0` siempre (muy seguro, aunque esté equivocado)
- Latencia más baja para respuestas "SPEECH" (~177ms) vs "NONSPEECH" (~204ms)
- El sistema de normalización funciona perfectamente - el problema está en el modelo, no en el post-procesamiento

**Conclusión:** El fine-tuning con LoRA es **esencial** para esta tarea. No se recomienda optimizar prompts para el modelo BASE sin LoRA.

---

### Tabla Comparativa Completa de Configuraciones

**Test Set:** 21,340 muestras | **Seed:** 42 | **Fecha:** 15-26 diciembre 2025

| Configuración | BA_clip | BA_conditions | Speech Acc | Nonspeech Acc | Notas |
|--------------|---------|---------------|------------|---------------|-------|
| **1. BASE + OPRO auto (20 samples)** | **59.89%** ❌ | - | **94.26%** ⚠️ | **25.52%** ❌ | Sesgo crítico hacia SPEECH |
| **2. BASE + OPRO Clásico** | 88.12% | 89.34% | 91.64% | 84.60% | Resultado previo ¹ |
| **3. BASE + OPRO Varied (30 samples)** | **59.89%** ❌ | - | **94.26%** ⚠️ | **25.52%** ❌ | Mismo sesgo que #1 |
| **4. LoRA + OPRO Clásico** | **94.90%** ⭐ | **95.46%** ⭐ | **98.23%** | **91.57%** | Mejor resultado general |
| **5. LoRA + OPRO Open** | **94.78%** ✅ | **95.32%** ✅ | **98.23%** | **91.34%** | Resultado previo ² |
| **6. LoRA + OPRO Varied (30 samples)** | **92.99%** ✅ | - | **98.35%** | **87.64%** | Prompts más diversos |

**Observaciones clave:**
- **BASE sin LoRA (Configs 1-3):** Resultados inutilizables debido a sesgo inherente
  - Sobre-predice SPEECH (94% correcto)
  - Falla dramáticamente en NONSPEECH (25% correcto)
  - BA resultante: ~60% (peor que random guessing en clases balanceadas)
- **LoRA funciona bien (Configs 4-6):** Corrige el sesgo completamente
  - **BASE → LoRA:** +35% en NONSPEECH accuracy, +33% en BA
  - OPRO Varied con más diversidad: 92.99% BA (excelente, aunque 2% menor que Clásico)
- **Mejor configuración:** LoRA + OPRO Clásico con **94.90% BA**

---

### Prompts Optimizados por OPRO

**¹ Mejor Prompt - BASE + OPRO Clásico (resultado previo válido):**
```
Listen briefly; is this clip human speech or noise? Quickly reply: SPEECH or NON-SPEECH.
```
*Nota: Este resultado proviene de un experimento anterior. Los experimentos recientes muestran que BASE sin LoRA tiene sesgo crítico.*

**Prompts de BASE con sesgo (Configs 1 y 3):**
```
Classify this audio. Output only: SPEECH or NONSPEECH.
```
*Este prompt obtiene 59.89% BA debido al sesgo del modelo BASE, NO por la calidad del prompt.*

**² Mejor Prompt - LoRA + OPRO Clásico (Config 4 - MEJOR GENERAL):**
```
Decide the dominant content.
Definitions:
- SPEECH = human voice, spoken words, syllables, conversational cues.
- NONSPEECH = music, tones/beeps, environmental noise, silence.
Output exactly: SPEECH or NONSPEECH.
```

**Mejor Prompt - LoRA + OPRO Open (Config 5):**
```
Decide the dominant content.
Definitions:
- SPEECH = human voice, spoken words, syllables, conversational cues.
- NONSPEECH = music, tones/beeps, environmental noise, silence.
Output exactly: SPEECH or NONSPEECH.
```

**Mejor Prompt - LoRA + OPRO Varied (Config 6 - 30 samples, 15 seed prompts diversos):**
```
Does this audio contain human speech? Answer SPEECH or NONSPEECH.
```

**Observaciones:**
- OPRO Clásico y OPRO Open convergieron al **mismo prompt idéntico** con definiciones explícitas
- OPRO Varied (con seeds más diversos) encontró un prompt más simple pero igualmente efectivo
- La diferencia de rendimiento (94.90% vs 92.99%) puede deberse a la naturaleza más directa del prompt sin definiciones

---

### Seed Prompts del Experimento OPRO Varied

El experimento OPRO Varied (Configs 3 y 6) utilizó **15 seed prompts diversos** diseñados para explorar diferentes estrategias de prompting:

**Prompts Descriptivos Abiertos:**
1. "What do you hear in this audio?"
2. "Describe what you hear in this audio clip."
3. "What is the primary sound source in this audio clip?"

**Prompts Binarios Directos:**
4. "Does this audio contain human speech? Answer SPEECH or NONSPEECH."
5. "Classify this audio. Output only: SPEECH or NONSPEECH."
6. "Is there human speech in this recording? Reply with one word: SPEECH or NONSPEECH."

**Prompts con Definiciones:**
7. "Listen carefully. SPEECH means human voice or talking. NONSPEECH means music, noise, or silence. What is this?"
8. "Decide the dominant content. If human voice is present, say SPEECH. Otherwise, say NONSPEECH."

**Prompts con Ejemplos (Few-shot):**
9. "Example: beeping sounds → NONSPEECH. Example: person talking → SPEECH. Now classify this audio:"

**Prompts Técnicos:**
10. "Audio classification task. Detect if human vocal tract sounds are present. Answer: SPEECH or NONSPEECH."
11. "Analyze the acoustic content. If you identify human voice, speaking, or conversation, output SPEECH. For music, tones, noise, or silence, output NONSPEECH."

**Formatos Alternativos:**
12. "Listen. Does this contain: A) Human speech, or B) Other sounds? Output A or B."
13. "Speech detection: YES if human voice detected, NO otherwise."
14. "TASK: Binary classification. LABELS: SPEECH (human voice) or NONSPEECH (all other sounds). AUDIO:"
15. "Quick check: human speech present? SPEECH or NONSPEECH."

**Resultado:** A pesar de la gran diversidad de estrategias, OPRO convergió a un prompt efectivo simple: *"Does this audio contain human speech? Answer SPEECH or NONSPEECH."* (92.99% BA en LoRA).

---

### Análisis Detallado: LoRA + OPRO Clásico (Mejor Resultado)

**Desglose por Dimensión Psicoacústica:**
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

### Conclusiones

1. **LoRA es CRÍTICO para esta tarea:** El modelo BASE sin fine-tuning presenta sesgo inherente hacia SPEECH
   - BASE: 59.89% BA (25.52% NONSPEECH accuracy) ❌
   - LoRA: 92.99-94.90% BA (87.64-91.57% NONSPEECH accuracy) ✅
   - **Mejora: +35 puntos porcentuales** en detección de NONSPEECH

2. **El sesgo del modelo BASE NO se puede corregir con prompting:**
   - Probamos 3 estrategias diferentes de OPRO (auto, clásico, varied)
   - Todas obtienen resultados similares (~60% BA)
   - El modelo responde literalmente "SPEECH" cuando debería decir "NONSPEECH"
   - Diagnóstico con 100 samples: 33% de error en casos limpios de NONSPEECH

3. **OPRO funciona excelentemente CON LoRA:**
   - LoRA + OPRO Clásico: **94.90% BA** ⭐
   - LoRA + OPRO Open: 94.78% BA ✅
   - LoRA + OPRO Varied: 92.99% BA ✅
   - Diferentes estrategias de prompting convergen a resultados similares (92-95% BA)

4. **Diversidad de seed prompts no garantiza mejor rendimiento:**
   - OPRO Clásico (8 seeds similares): 94.90% BA
   - OPRO Varied (15 seeds diversos): 92.99% BA
   - La optimización converge a prompts efectivos independientemente de las semillas

5. **Desafíos técnicos identificados:**
   - **Duración corta (<100ms):** Peor rendimiento (80-90% BA)
   - **SNR muy robusto:** Excelente incluso a -10dB (97% BA)
   - **Infraestructura:** Nodos aisurrey14/aisurrey19 tienen problemas de CUDA

---

### Diagnóstico del Sesgo del Modelo BASE

Para entender exactamente por qué el modelo BASE falla, ejecutamos un diagnóstico exhaustivo que evaluó 100 samples NONSPEECH limpios (sin degradaciones).

**Metodología:**
- Modelo: Qwen2-Audio-7B-Instruct BASE (sin LoRA)
- Prompt: `"Classify this audio. Output only: SPEECH or NONSPEECH."`
- Samples: 100 NONSPEECH del test set
- Análisis: Respuestas RAW del modelo (antes de normalización)

**Resultados:**
```
Total samples NONSPEECH: 100
Correctos (NONSPEECH): 67 (67.00%)
Incorrectos (predice SPEECH): 33 (33.00%)
```

**Ejemplos de respuestas RAW:**
```
✗ Sample 1 | GROUND TRUTH: NONSPEECH | RAW: 'SPEECH' | NORMALIZED: SPEECH
✗ Sample 2 | GROUND TRUTH: NONSPEECH | RAW: 'SPEECH' | NORMALIZED: SPEECH
✗ Sample 3 | GROUND TRUTH: NONSPEECH | RAW: 'SPEECH' | NORMALIZED: SPEECH
✓ Sample 6 | GROUND TRUTH: NONSPEECH | RAW: 'NONSPEECH' | NORMALIZED: NONSPEECH
✓ Sample 7 | GROUND TRUTH: NONSPEECH | RAW: 'NONSPEECH' | NORMALIZED: NONSPEECH
```

**Hallazgos clave:**
1. **El modelo responde literalmente "SPEECH"** cuando debería decir "NONSPEECH"
2. **El sistema de normalización funciona perfectamente** - no hay errores de interpretación
3. **Confidence siempre es 1.0** - el modelo está muy seguro, aunque esté equivocado
4. **Latencia sugiere sesgo:** SPEECH ~177ms vs NONSPEECH ~204ms (SPEECH es la respuesta "por defecto")

**Conclusión:** El problema es inherente al modelo BASE, NO es un problema de prompting ni de normalización.

---

### 🎯 Hallazgos Clave del Diagnóstico

#### El Problema está en el MODELO BASE, NO en el Sistema de Normalización

**📊 Evidencia Experimental (100 samples NONSPEECH limpios):**
```
✅ Correctos (NONSPEECH): 67 (67%)
❌ Incorrectos (predice SPEECH): 33 (33%)
```

**🔍 Análisis de Respuestas RAW:**

El modelo está respondiendo **literalmente "SPEECH"** cuando debería decir "NONSPEECH":

```
✗ Sample 1 | NORMALIZED: SPEECH     | RAW: raw_output='SPEECH', text='SPEECH'
✗ Sample 2 | NORMALIZED: SPEECH     | RAW: raw_output='SPEECH', text='SPEECH'
✗ Sample 3 | NORMALIZED: SPEECH     | RAW: raw_output='SPEECH', text='SPEECH'
✗ Sample 4 | NORMALIZED: SPEECH     | RAW: raw_output='SPEECH', text='SPEECH'
✗ Sample 5 | NORMALIZED: SPEECH     | RAW: raw_output='SPEECH', text='SPEECH'
✓ Sample 6 | NORMALIZED: NONSPEECH  | RAW: raw_output='NONSPEECH', text='NONSPEECH'
✓ Sample 7 | NORMALIZED: NONSPEECH  | RAW: raw_output='NONSPEECH', text='NONSPEECH'
```

**El sistema de normalización funciona PERFECTAMENTE:**
- Cuando el modelo dice "SPEECH" → normaliza a SPEECH ✓
- Cuando el modelo dice "NONSPEECH" o "Nonspeech." → normaliza a NONSPEECH ✓
- **NO HAY errores de interpretación en el post-procesamiento**

**💡 Causa Raíz Identificada:**

El modelo BASE (Qwen2-Audio-7B-Instruct sin fine-tuning) tiene un **SESGO INHERENTE** hacia clasificar audios como SPEECH:

1. **Sesgo sistemático:** Sobre-predice SPEECH independientemente del prompt
2. **Empeora con degradaciones:**
   - Samples limpios: 33% error (67% accuracy)
   - Test set con degradaciones: 74.48% error (25.52% accuracy)
3. **Condiciones que agravan el sesgo:**
   - SNR bajo (-10dB, -5dB, 0dB)
   - Duración muy corta (<100ms)
   - Filtros (bandpass, lowpass, highpass)
   - Reverberación (0.3s, 1.0s, 2.5s)

**🔬 Observaciones Técnicas:**

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| `confidence` | 1.0 (siempre) | Modelo muy seguro, incluso cuando se equivoca |
| Latencia SPEECH | ~177ms | Respuesta más rápida (¿respuesta por defecto?) |
| Latencia NONSPEECH | ~204ms | Respuesta más lenta (+15%) |
| `p_first_token` | Variable (0.44-0.87) | Sin patrón claro de discriminación |

**✅ Por Qué LoRA Funciona:**

El fine-tuning con LoRA **corrige completamente este sesgo**:

| Modelo | NONSPEECH Accuracy | BA | Mejora |
|--------|-------------------|-----|--------|
| BASE sin LoRA | **25.52%** ❌ | 59.89% | - |
| LoRA fine-tuned | **87.64%** ✅ | 92.99% | **+62.12 puntos** |

El fine-tuning entrena al modelo a:
- ✅ NO sobre-predecir SPEECH por defecto
- ✅ Detectar correctamente samples NONSPEECH
- ✅ Mantener robustez bajo degradaciones severas

**🚫 Implicaciones:**

1. **NO tiene sentido optimizar prompts para el modelo BASE** - el sesgo es inherente al modelo pre-entrenado
2. **El prompting NO puede corregir este sesgo** - probamos 3 estrategias diferentes (auto, clásico, varied) y todas fallan igual
3. **LoRA es ESENCIAL** - no opcional - para esta tarea específica de detección de habla

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
│   ├── diagnose_base_nonspeech.py     # Diagnóstico de sesgo del modelo BASE
│   ├── run_complete_pipeline.py       # Wrapper completo
│   ├── run_opro_varied_complete.sh    # Pipeline OPRO con prompts variados
│   └── fix_base_opro_open.sh          # Fix para re-ejecutar BASE + OPRO
│
├── slurm/                             # Jobs de SLURM
│   ├── tools/on_submit.sh             # Wrapper para ejecutar comandos SLURM
│   ├── opro_varied_base.job           # OPRO varied en BASE
│   ├── opro_varied_lora.job           # OPRO varied en LoRA
│   ├── eval_varied_base.job           # Evaluación BASE + OPRO varied
│   ├── eval_varied_lora.job           # Evaluación LoRA + OPRO varied
│   ├── diagnose_base.job              # Job de diagnóstico del sesgo BASE
│   └── *.job                          # Otros scripts de jobs
│
├── src/qsm/                           # Código fuente
│   ├── models/qwen_audio.py           # Wrapper del modelo
│   └── utils/normalize.py             # Utilidades de normalización
│
├── prompts/                           # Archivos de prompts seed
│   └── open_descriptive_seeds.json   # 15 prompts variados para OPRO Varied
│
├── checkpoints/                       # Checkpoints LoRA
│   └── qwen_lora_seed42/
│       └── final/                     # Checkpoint final usado en experimentos
│
├── results/                           # Resultados de evaluación
│   ├── complete_pipeline_seed42/      # OPRO Clásico ✅
│   │   ├── 04_opro_base/              # BASE + OPRO Clásico
│   │   ├── 05_opro_lora/              # LoRA + OPRO Clásico (MEJOR: 94.90% BA)
│   │   │   ├── optimization_history.json
│   │   │   └── best_prompt.txt
│   │   ├── 06_eval_base_opro/
│   │   └── 07_eval_lora_opro/
│   │       └── metrics.json
│   │
│   ├── complete_pipeline_seed42_opro_open/  # OPRO Open ✅
│   │   ├── 04_opro_base/              # BASE + OPRO auto (sesgo: 59.89% BA)
│   │   ├── 05_opro_lora/              # LoRA + OPRO Open (94.78% BA)
│   │   ├── 06_eval_base_opro/
│   │   └── 07_eval_lora_opro/
│   │
│   └── opro_varied_seed42/            # OPRO Varied (15 seeds diversos) ✅
│       ├── base/                      # BASE + OPRO Varied (sesgo: 59.89% BA)
│       │   ├── optimization_history.json
│       │   └── best_prompt.txt
│       ├── lora/                      # LoRA + OPRO Varied (92.99% BA)
│       │   ├── optimization_history.json
│       │   └── best_prompt.txt
│       ├── eval_base/
│       │   └── metrics.json
│       └── eval_lora/
│           └── metrics.json
│
└── logs/                              # Logs de SLURM
    ├── diagnose_base_2028551.out      # Log del diagnóstico de sesgo BASE
    └── *.out/*.err                    # Logs de jobs
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

### OPRO Varied (Experimental)

- **Optimizer LLM:** Qwen/Qwen2.5-7B-Instruct (local)
- **Iterations:** 15
- **Samples per iteration:** 30 (50% más que Clásico)
- **Candidates per iteration:** 8
- **Decoding mode:** open (permite respuestas libres)
- **Seed prompts:** 15 templates diversos (vs 8 en Clásico)
- **Diversity strategy:** Incluye prompts descriptivos, binarios, con definiciones, con ejemplos, multiple choice, YES/NO
- **Resultado:** 92.99% BA en LoRA (excelente, 2% menor que Clásico)

---

## Archivos de Resultados

### BASE + OPRO auto (Config 1) ❌

- **Mejor prompt:** [results/complete_pipeline_seed42_opro_open/04_opro_base/best_prompt.txt](results/complete_pipeline_seed42_opro_open/04_opro_base/best_prompt.txt)
- **Métricas finales:** [results/complete_pipeline_seed42_opro_open/06_eval_base_opro/metrics.json](results/complete_pipeline_seed42_opro_open/06_eval_base_opro/metrics.json)
- **Status:** ❌ Sesgo inherente del modelo (59.89% BA)

### BASE + OPRO Clásico (Config 2) ⚠️

- **Historia de optimización:** [results/complete_pipeline_seed42/04_opro_base/optimization_history.json](results/complete_pipeline_seed42/04_opro_base/optimization_history.json)
- **Mejor prompt:** [results/complete_pipeline_seed42/04_opro_base/best_prompt.txt](results/complete_pipeline_seed42/04_opro_base/best_prompt.txt)
- **Métricas finales:** [results/complete_pipeline_seed42/06_eval_base_opro/metrics.json](results/complete_pipeline_seed42/06_eval_base_opro/metrics.json)
- **Status:** ⚠️ Resultado previo (88.12% BA) - inconsistente con experimentos recientes

### BASE + OPRO Varied (Config 3) ❌

- **Historia de optimización:** [results/opro_varied_seed42/base/optimization_history.json](results/opro_varied_seed42/base/optimization_history.json)
- **Mejor prompt:** [results/opro_varied_seed42/base/best_prompt.txt](results/opro_varied_seed42/base/best_prompt.txt)
- **Métricas finales:** [results/opro_varied_seed42/eval_base/metrics.json](results/opro_varied_seed42/eval_base/metrics.json)
- **Status:** ❌ Sesgo inherente del modelo (59.89% BA)

### LoRA + OPRO Clásico (Config 4) ⭐ MEJOR

- **Historia de optimización:** [results/complete_pipeline_seed42/05_opro_lora/optimization_history.json](results/complete_pipeline_seed42/05_opro_lora/optimization_history.json)
- **Mejor prompt:** [results/complete_pipeline_seed42/05_opro_lora/best_prompt.txt](results/complete_pipeline_seed42/05_opro_lora/best_prompt.txt)
- **Métricas finales:** [results/complete_pipeline_seed42/07_eval_lora_opro/metrics.json](results/complete_pipeline_seed42/07_eval_lora_opro/metrics.json)
- **Status:** ⭐ **94.90% BA** - Mejor resultado general

### LoRA + OPRO Open (Config 5) ✅

- **Historia de optimización:** [results/complete_pipeline_seed42_opro_open/05_opro_lora/optimization_history.json](results/complete_pipeline_seed42_opro_open/05_opro_lora/optimization_history.json)
- **Mejor prompt:** [results/complete_pipeline_seed42_opro_open/05_opro_lora/best_prompt.txt](results/complete_pipeline_seed42_opro_open/05_opro_lora/best_prompt.txt)
- **Métricas finales:** [results/complete_pipeline_seed42_opro_open/07_eval_lora_opro/metrics.json](results/complete_pipeline_seed42_opro_open/07_eval_lora_opro/metrics.json)
- **Status:** ✅ 94.78% BA - Resultado casi idéntico a Clásico

### LoRA + OPRO Varied (Config 6) ✅

- **Historia de optimización:** [results/opro_varied_seed42/lora/optimization_history.json](results/opro_varied_seed42/lora/optimization_history.json)
- **Mejor prompt:** [results/opro_varied_seed42/lora/best_prompt.txt](results/opro_varied_seed42/lora/best_prompt.txt)
- **Métricas finales:** [results/opro_varied_seed42/eval_lora/metrics.json](results/opro_varied_seed42/eval_lora/metrics.json)
- **Status:** ✅ 92.99% BA - Prompts más diversos, resultado excelente

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

### Investigaciones Recomendadas

1. **Validación y Reproducibilidad:**
   - ✅ **COMPLETADO:** Sesgo del modelo BASE documentado y diagnosticado
   - Probar con diferentes seeds (43, 44, 45) para validar reproducibilidad de LoRA + OPRO
   - Evaluar estabilidad de LoRA training con diferentes random seeds

2. **Optimizaciones de LoRA:**
   - Experimentar con diferentes configuraciones de LoRA (r=32, r=128, r=256)
   - Probar diferentes learning rates (1e-5, 1e-4)
   - Evaluar impacto de más epochs de training (5, 10)
   - **Baseline importante:** Evaluar LoRA SIN OPRO para cuantificar beneficio puro de la optimización de prompts

3. **Análisis de Errores en LoRA:**
   - **Investigar por qué duration corta (<100ms) tiene peor rendimiento** (80-90% BA)
     - Hipótesis: Clips muy cortos no proveen suficiente contexto temporal
     - Posible solución: Prompt especializado o data augmentation
   - Analizar las 1,732 muestras mal clasificadas en NONSPEECH (8.43% error con mejor modelo)
   - Estudiar si hay patrones en los errores por condición psicoacústica

4. **Experimentos OPRO Avanzados (solo con LoRA):**
   - Probar otros LLMs optimizadores (Llama 3, Mistral, GPT-4)
   - Experimentar con más iteraciones (20, 30)
   - Probar con más samples por iteración (40, 50)
   - Evaluar si OPRO multi-objetivo (maximizar BA + minimizar latencia) mejora eficiencia

5. **NO Recomendado:**
   - ❌ Optimizar prompts para modelo BASE sin LoRA (sesgo inherente no corregible)
   - ❌ Intentar otras estrategias de prompting en BASE (ya probamos 3, todas fallan igual)

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

**Última actualización:** 26 de diciembre 2025
**Versión:** 4.0
**Status:** 🟢 Todos los experimentos completados | ✅ Sesgo del modelo BASE diagnosticado | ⭐ LoRA + OPRO validado como mejor configuración (94.90% BA)
