# MANIFEST: Repositorio Limpio OPRO Qwen

**Generado:** 2025-12-11
**Versión:** 1.0

---

## Resumen

- **Directorio:** `c:\VS projects\opro2_clean\`
- **Dataset:** `c:\VS projects\opro2\data\` (referenciado, NO copiado)
- **Total archivos:** 23 (13 copiados + 10 nuevos)
- **Scripts core:** 4 principales + 1 wrapper
- **SLURM jobs:** 7 (uno por etapa)

---

## Inventario por Categoría

### Configuración (4 archivos)

| Archivo | Origen | Modificado | Descripción |
|---------|--------|------------|-------------|
| `requirements.txt` | Copiado | No | Dependencias Python |
| `pyproject.toml` | Copiado | No | Build config |
| `config.yaml` | Copiado | Sí (rutas) | Configuración global, apunta a `c:/VS projects/opro2/data` |
| `configs/pipeline_config.yaml` | Copiado | Sí (rutas) | Configuración del pipeline, rutas a opro2/data |

### Código Fuente (5 archivos)

| Archivo | Origen | Modificado | Líneas | Descripción |
|---------|--------|------------|--------|-------------|
| `src/qsm/__init__.py` | Copiado | No | - | Inicialización del paquete |
| `src/qsm/models/__init__.py` | Copiado | No | - | Inicialización de módulo models |
| `src/qsm/models/qwen_audio.py` | Copiado | No | ~450 | Wrapper Qwen2AudioClassifier |
| `src/qsm/utils/__init__.py` | Nuevo | - | 1 | Inicialización de módulo utils |
| `src/qsm/utils/normalize.py` | Copiado | No | ~120 | Normalización de respuestas |

### Scripts Core Pipeline (5 archivos)

| Archivo | Origen | Modificado | Etapas | Líneas | Descripción |
|---------|--------|------------|--------|--------|-------------|
| `scripts/evaluate_simple.py` | Copiado | Sí (+--prompt_file) | 1,3,6,7 | ~250 | Evaluación principal con métricas por condición |
| `scripts/finetune_qwen_audio.py` | Copiado | No | 2 | ~450 | Entrenamiento LoRA con QLoRA |
| `scripts/opro_classic_optimize.py` | Copiado | No | 4,5 | ~1,295 | Optimización OPRO con LLM local |
| `scripts/opro_post_ft_v2.py` | Copiado | No | 5 | ~570 | OPRO post-finetuning (templates) |
| `scripts/run_complete_pipeline.py` | Nuevo | - | 1-7 | ~280 | Wrapper completo del pipeline de 7 etapas |

### SLURM Jobs (7 archivos)

| Archivo | Tipo | Etapa | Script que ejecuta | Partición | Tiempo | Memoria |
|---------|------|-------|-------------------|-----------|--------|---------|
| `slurm/01_finetune_lora.job` | Nuevo | 2 | finetune_qwen_audio.py | 3090 | 12h | 64GB |
| `slurm/02_opro_base.job` | Nuevo | 4 | opro_classic_optimize.py --no_lora | debug | 4h | 48GB |
| `slurm/03_opro_lora.job` | Nuevo | 5 | opro_post_ft_v2.py | debug | 2.5h | 48GB |
| `slurm/04_eval_base.job` | Nuevo | 3a | evaluate_simple.py | 3090 | 2h | 48GB |
| `slurm/05_eval_lora.job` | Nuevo | 3b | evaluate_simple.py --checkpoint | 3090 | 2h | 48GB |
| `slurm/06_eval_base_opro.job` | Nuevo | 6 | evaluate_simple.py --prompt_file | 3090 | 2h | 48GB |
| `slurm/07_eval_lora_opro.job` | Nuevo | 7 | evaluate_simple.py --checkpoint --prompt_file | 3090 | 2h | 48GB |

### Documentación (2 archivos)

| Archivo | Tipo | Descripción |
|---------|------|-------------|
| `README.md` | Nuevo | Documentación principal del pipeline |
| `MANIFEST.md` | Nuevo | Este archivo - inventario completo |

### Otros (2 archivos)

| Archivo | Descripción |
|---------|-------------|
| `checkpoints/.gitkeep` | Placeholder para checkpoints LoRA |
| `results/.gitkeep` | Placeholder para resultados de evaluación |

---

## Mapa de Dependencias por Etapa

### Etapa 1: Evaluación Psicoacústica (Baseline)

```
evaluate_simple.py
  ├─ src/qsm/models/qwen_audio.py
  │   └─ transformers, torch, librosa, soundfile
  ├─ src/qsm/utils/normalize.py
  └─ c:\VS projects\opro2\data\processed\conditions_final\conditions_manifest_split.parquet
```

**Output:** `results/eval_base_baseline/`

---

### Etapa 2: LoRA Finetuning

```
finetune_qwen_audio.py
  ├─ transformers (Qwen2-Audio-7B-Instruct)
  ├─ peft (LoRA config, get_peft_model)
  ├─ bitsandbytes (4-bit quantization)
  └─ c:\VS projects\opro2\data\processed\normalized_clips\
      ├─ train_metadata.csv
      └─ test_metadata.csv
```

**Output:** `checkpoints/qwen_lora_seed42/final/`
- `adapter_model.safetensors` (pesos LoRA)
- `adapter_config.json` (configuración PEFT)

---

### Etapa 3: Evaluación Base vs LoRA

**3a: BASE model**
```
evaluate_simple.py (sin checkpoint)
  └─ Output: results/eval_base/
```

**3b: LoRA model**
```
evaluate_simple.py --checkpoint
  ├─ checkpoints/qwen_lora_seed42/final/
  └─ Output: results/eval_lora/
```

---

### Etapa 4: OPRO Base

```
opro_classic_optimize.py --no_lora
  ├─ Qwen/Qwen2.5-7B-Instruct (optimizer LLM)
  ├─ src/qsm/models/qwen_audio.py (evaluator model)
  └─ c:\VS projects\opro2\data\processed\conditions_final\conditions_manifest_split.parquet (dev split)
```

**Output:** `results/opro_base_seed42/`
- `best_prompt.txt` - Mejor prompt optimizado
- `opro_history.json` - Curva de recompensa
- `opro_memory.json` - Top-k prompts

---

### Etapa 5: OPRO LoRA

```
opro_post_ft_v2.py --checkpoint
  ├─ checkpoints/qwen_lora_seed42/final/
  ├─ src/qsm/models/qwen_audio.py
  └─ c:\VS projects\opro2\data\processed\experimental_variants\dev_metadata.csv
```

**Output:** `results/opro_lora_seed42/`
- `best_prompt.txt` - Mejor prompt para LoRA
- `optimization_history.json`

---

### Etapa 6: Evaluación BASE + OPRO

```
evaluate_simple.py --prompt_file
  ├─ results/opro_base_seed42/best_prompt.txt
  ├─ src/qsm/models/qwen_audio.py (BASE model)
  └─ c:\VS projects\opro2\data\processed\conditions_final\conditions_manifest_split.parquet
```

**Output:** `results/eval_base_opro/`
**Expected BA:** ~86.9%

---

### Etapa 7: Evaluación LoRA + OPRO

```
evaluate_simple.py --checkpoint --prompt_file
  ├─ checkpoints/qwen_lora_seed42/final/
  ├─ results/opro_lora_seed42/best_prompt.txt
  ├─ src/qsm/models/qwen_audio.py
  └─ c:\VS projects\opro2\data\processed\conditions_final\conditions_manifest_split.parquet
```

**Output:** `results/eval_lora_opro/`
**Expected BA:** ~93.7% ⭐ **BEST RESULT**

---

## Comandos de Ejemplo

### Ejecución Local (Python directo)

```bash
# Etapa 1: Baseline evaluation
python scripts/evaluate_simple.py \
  --manifest "c:/VS projects/opro2/data/processed/conditions_final/conditions_manifest_split.parquet" \
  --prompt "Does this audio contain human speech? Answer SPEECH or NONSPEECH." \
  --output_dir results/eval_baseline \
  --batch_size 50

# Etapa 2: LoRA training
python scripts/finetune_qwen_audio.py \
  --train_csv "c:/VS projects/opro2/data/processed/normalized_clips/train_metadata.csv" \
  --val_csv "c:/VS projects/opro2/data/processed/normalized_clips/test_metadata.csv" \
  --output_dir checkpoints/qwen_lora_seed42 \
  --seed 42 \
  --num_epochs 3

# Etapa 3a: Eval BASE
python scripts/evaluate_simple.py \
  --manifest "c:/VS projects/opro2/data/processed/conditions_final/conditions_manifest_split.parquet" \
  --prompt "Does this audio contain human speech? Answer SPEECH or NONSPEECH." \
  --output_dir results/eval_base \
  --batch_size 50

# Etapa 3b: Eval LoRA
python scripts/evaluate_simple.py \
  --manifest "c:/VS projects/opro2/data/processed/conditions_final/conditions_manifest_split.parquet" \
  --prompt "Does this audio contain human speech? Answer SPEECH or NONSPEECH." \
  --checkpoint checkpoints/qwen_lora_seed42/final \
  --output_dir results/eval_lora \
  --batch_size 50

# Etapa 4: OPRO BASE
python scripts/opro_classic_optimize.py \
  --manifest "c:/VS projects/opro2/data/processed/conditions_final/conditions_manifest_split.parquet" \
  --split dev \
  --output_dir results/opro_base_seed42 \
  --no_lora \
  --num_iterations 30 \
  --seed 42

# Etapa 5: OPRO LoRA
python scripts/opro_post_ft_v2.py \
  --checkpoint checkpoints/qwen_lora_seed42/final \
  --train_csv "c:/VS projects/opro2/data/processed/experimental_variants/dev_metadata.csv" \
  --output_dir results/opro_lora_seed42 \
  --num_iterations 15

# Etapa 6: Eval BASE + OPRO
python scripts/evaluate_simple.py \
  --manifest "c:/VS projects/opro2/data/processed/conditions_final/conditions_manifest_split.parquet" \
  --prompt_file results/opro_base_seed42/best_prompt.txt \
  --output_dir results/eval_base_opro \
  --batch_size 50

# Etapa 7: Eval LoRA + OPRO
python scripts/evaluate_simple.py \
  --manifest "c:/VS projects/opro2/data/processed/conditions_final/conditions_manifest_split.parquet" \
  --prompt_file results/opro_lora_seed42/best_prompt.txt \
  --checkpoint checkpoints/qwen_lora_seed42/final \
  --output_dir results/eval_lora_opro \
  --batch_size 50
```

### Ejecución en SLURM (secuencial)

```bash
# Ejecutar las 7 etapas secuencialmente
cd /mnt/fast/nobackup/users/gb0048/opro2_clean

# Etapa 1: Baseline (opcional, puede omitirse)
# (evaluación simple con prompt base, no requiere job de SLURM)

# Etapa 2: Training (12h)
sbatch slurm/01_finetune_lora.job 42
# Esperar a que termine...

# Etapas 3a y 3b: Evaluación comparativa (4h total)
sbatch slurm/04_eval_base.job
sbatch slurm/05_eval_lora.job 42
# Pueden ejecutarse en paralelo

# Etapa 4: OPRO BASE (4h)
sbatch slurm/02_opro_base.job 42

# Etapa 5: OPRO LoRA (2.5h)
sbatch slurm/03_opro_lora.job 42

# Etapa 6: Eval BASE + OPRO (2h)
sbatch slurm/06_eval_base_opro.job 42

# Etapa 7: Eval LoRA + OPRO (2h)
sbatch slurm/07_eval_lora_opro.job 42

# Tiempo total: ~26-28 horas (si se ejecuta secuencialmente)
```

### Ejecución con Wrapper (automática)

```bash
python scripts/run_complete_pipeline.py \
  --seed 42 \
  --data_root "c:/VS projects/opro2/data" \
  --output_dir results/pipeline_run_20251211

# Con opciones de skip
python scripts/run_complete_pipeline.py \
  --seed 42 \
  --skip_training \  # Usar checkpoint existente
  --skip_opro_base   # Usar prompt existente
```

---

## Referencias a Dataset Original

Todos los scripts apuntan al dataset original en `opro2`:

**Manifests:**
- `c:/VS projects/opro2/data/processed/conditions_final/conditions_manifest_split.parquet` (1,740 samples: 1,400 dev + 340 test)
- `c:/VS projects/opro2/data/processed/experimental_variants/dev_metadata.csv`
- `c:/VS projects/opro2/data/processed/normalized_clips/train_metadata.csv`

**Audio files:**
- `c:/VS projects/opro2/data/processed/conditions_final/{duration,snr,band,rir}/*.wav`

**NO se copia ningún archivo de datos.** El dataset original permanece en `opro2/`.

---

## Estructura Completa de Archivos

```
opro2_clean/
├── README.md                          [Documentación principal]
├── MANIFEST.md                        [Este archivo]
├── requirements.txt                   [Dependencias Python]
├── pyproject.toml                     [Build config]
├── config.yaml                        [Config global - rutas modificadas]
│
├── configs/
│   └── pipeline_config.yaml           [Config pipeline - rutas modificadas]
│
├── src/
│   └── qsm/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── qwen_audio.py          [Wrapper Qwen2AudioClassifier]
│       └── utils/
│           ├── __init__.py
│           └── normalize.py           [Normalización de respuestas]
│
├── scripts/
│   ├── evaluate_simple.py             [Modificado: +flag --prompt_file]
│   ├── finetune_qwen_audio.py         [LoRA training]
│   ├── opro_classic_optimize.py       [OPRO con LLM local]
│   ├── opro_post_ft_v2.py             [OPRO post-FT]
│   └── run_complete_pipeline.py       [Wrapper completo - Nuevo]
│
├── slurm/
│   ├── 01_finetune_lora.job           [Etapa 2]
│   ├── 02_opro_base.job               [Etapa 4]
│   ├── 03_opro_lora.job               [Etapa 5]
│   ├── 04_eval_base.job               [Etapa 3a]
│   ├── 05_eval_lora.job               [Etapa 3b]
│   ├── 06_eval_base_opro.job          [Etapa 6]
│   └── 07_eval_lora_opro.job          [Etapa 7]
│
├── checkpoints/
│   └── .gitkeep                       [Placeholder - checkpoints LoRA irán aquí]
│
└── results/
    └── .gitkeep                       [Placeholder - resultados irán aquí]
```

---

## Métricas y Resultados Esperados

| Configuración | BA_clip | Mejora sobre Baseline | Etapa |
|---------------|---------|----------------------|-------|
| BASE (baseline) | ~80% | - | 1 |
| BASE (evaluado) | ~82% | +2% | 3a |
| LoRA | ~88% | +8% | 3b |
| BASE + OPRO | **~86.9%** | +6.9% | 6 |
| **LoRA + OPRO** | **~93.7%** ⭐ | **+13.7%** | 7 |

**Métricas calculadas:**
- `BA_clip`: Balanced Accuracy sobre todos los clips
- `BA_conditions`: Mean BA sobre las 4 dimensiones psicoacústicas (duration, SNR, filter, reverb)
- Per-condition accuracy: 22 condiciones independientes

---

## Notas de Implementación

### Modificaciones Realizadas

1. **`config.yaml`**: Rutas de datos cambiadas a `c:/VS projects/opro2/data/`
2. **`configs/pipeline_config.yaml`**: Todas las rutas apuntan al dataset original, results/checkpoints a opro2_clean
3. **`scripts/evaluate_simple.py`**: Agregado argumento `--prompt_file` para leer prompts desde archivo
4. **`src/qsm/utils/__init__.py`**: Creado (no existía en original)

### Scripts Wrapper y SLURM Jobs

- `run_complete_pipeline.py`: Nuevo wrapper Python para ejecutar las 7 etapas
- 7 SLURM jobs: Creados desde cero basándose en jobs exitosos del repo original
- Recursos asignados basados en experiencia previa (3090 partition, 48-64GB RAM)

---

**Fin del MANIFEST**
