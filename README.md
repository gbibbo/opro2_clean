# OPRO Qwen - Pipeline Limpio

Pipeline completo de optimización de prompts (OPRO) para detección de habla con Qwen2-Audio y LoRA.

---

## Descripción

Este repositorio contiene SOLO los archivos esenciales para ejecutar el pipeline de 7 etapas que optimiza la detección de habla mediante:
- **OPRO (Optimization by PROmpting)**: Optimización automática de prompts usando un LLM local
- **LoRA (Low-Rank Adaptation)**: Fine-tuning eficiente del modelo Qwen2-Audio-7B-Instruct
- **Evaluación psicoacústica**: Medición de rendimiento bajo 22 condiciones independientes

**Resultado final:** 93.7% BA (Balanced Accuracy) ⭐

---

## Estructura del Proyecto

Este repositorio está organizado para ejecutar el pipeline de **7 etapas**:

1. **Evaluación psicoacústica (baseline)** - Establece la línea base
2. **LoRA finetuning** - Entre adaptadores sobre el modelo base
3. **Evaluación base vs LoRA** - Compara ambos modelos
4. **OPRO en modelo base** - Optimiza prompts para el modelo base
5. **OPRO en modelo LoRA** - Re-optimiza prompts para el modelo fine-tuned
6. **Evaluación base + OPRO** - Eval con prompts optimizados (~86.9% BA)
7. **Evaluación LoRA + OPRO** - Eval final con LoRA + OPRO (~93.7% BA) ⭐

---

## Requisitos

### Sistema
- Python >= 3.10
- CUDA >= 11.8
- **GPU:** 40GB+ VRAM para training, 24GB para inference (recomendado: RTX 3090 o A6000)
- RAM: 48-64GB
- Disco: 100GB+ libres

### Dataset
El dataset se encuentra en: **`c:/VS projects/opro2/data/`**

**IMPORTANTE:** Este repositorio NO incluye los archivos de audio. Solo contiene código y configuración. Los scripts apuntan al dataset original en `opro2/data/`.

---

## Instalación

### 1. Clonar o ubicarse en el repositorio
```bash
cd "c:/VS projects/opro2_clean"
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

**Dependencias principales:**
- `torch>=2.0.0`, `torchaudio>=2.0.0`
- `transformers>=4.40.0`
- `peft>=0.10.0` (LoRA)
- `pandas>=2.0.0`, `pyarrow>=15.0.0`
- `librosa>=0.10.1`, `soundfile>=0.12.1`
- `scipy`, `scikit-learn`, `matplotlib`

### 3. Verificar GPU
```bash
nvidia-smi
```

---

## Uso Rápido

### Opción 1: Wrapper Completo (Recomendado)

Ejecuta las 7 etapas automáticamente:

```bash
python scripts/run_complete_pipeline.py --seed 42
```

**Con opciones de skip:**
```bash
python scripts/run_complete_pipeline.py \
  --seed 42 \
  --skip_training \      # Usar checkpoint LoRA existente
  --skip_opro_base \     # Usar prompt BASE existente
  --output_dir results/my_run
```

---

### Opción 2: Etapas Individuales (SLURM)

Para ejecución en cluster con SLURM:

```bash
cd /mnt/fast/nobackup/users/gb0048/opro2_clean

# Etapa 2: LoRA Training (12h, requiere 64GB RAM + GPU)
sbatch slurm/01_finetune_lora.job 42

# Etapa 3a y 3b: Evaluación BASE vs LoRA (4h total)
sbatch slurm/04_eval_base.job
sbatch slurm/05_eval_lora.job 42

# Etapa 4: OPRO en BASE (4h)
sbatch slurm/02_opro_base.job 42

# Etapa 5: OPRO en LoRA (2.5h)
sbatch slurm/03_opro_lora.job 42

# Etapa 6: Eval BASE + OPRO (2h)
sbatch slurm/06_eval_base_opro.job 42

# Etapa 7: Eval LoRA + OPRO (2h) - MEJOR RESULTADO
sbatch slurm/07_eval_lora_opro.job 42
```

**Tiempo total:** ~26-28 horas (secuencial)

---

### Opción 3: Manual (Python directo)

#### Etapa 1: Evaluación Baseline
```bash
python scripts/evaluate_simple.py \
  --manifest "c:/VS projects/opro2/data/processed/conditions_final/conditions_manifest_split.parquet" \
  --prompt "Does this audio contain human speech? Answer SPEECH or NONSPEECH." \
  --output_dir results/eval_baseline \
  --batch_size 50
```

#### Etapa 2: LoRA Training
```bash
python scripts/finetune_qwen_audio.py \
  --train_csv "c:/VS projects/opro2/data/processed/normalized_clips/train_metadata.csv" \
  --val_csv "c:/VS projects/opro2/data/processed/normalized_clips/test_metadata.csv" \
  --output_dir checkpoints/qwen_lora_seed42 \
  --seed 42 \
  --num_epochs 3 \
  --lora_r 64 \
  --lora_alpha 16
```

#### Etapa 3a: Eval BASE
```bash
python scripts/evaluate_simple.py \
  --manifest "c:/VS projects/opro2/data/processed/conditions_final/conditions_manifest_split.parquet" \
  --prompt "Does this audio contain human speech? Answer SPEECH or NONSPEECH." \
  --output_dir results/eval_base \
  --batch_size 50
```

#### Etapa 3b: Eval LoRA
```bash
python scripts/evaluate_simple.py \
  --manifest "c:/VS projects/opro2/data/processed/conditions_final/conditions_manifest_split.parquet" \
  --prompt "Does this audio contain human speech? Answer SPEECH or NONSPEECH." \
  --checkpoint checkpoints/qwen_lora_seed42/final \
  --output_dir results/eval_lora \
  --batch_size 50
```

#### Etapa 4: OPRO en BASE
```bash
python scripts/opro_classic_optimize.py \
  --manifest "c:/VS projects/opro2/data/processed/conditions_final/conditions_manifest_split.parquet" \
  --split dev \
  --output_dir results/opro_base_seed42 \
  --no_lora \
  --num_iterations 30 \
  --candidates_per_iter 3 \
  --seed 42
```

#### Etapa 5: OPRO en LoRA
```bash
python scripts/opro_post_ft_v2.py \
  --checkpoint checkpoints/qwen_lora_seed42/final \
  --train_csv "c:/VS projects/opro2/data/processed/experimental_variants/dev_metadata.csv" \
  --output_dir results/opro_lora_seed42 \
  --num_iterations 15 \
  --samples_per_iter 20
```

#### Etapa 6: Eval BASE + OPRO
```bash
python scripts/evaluate_simple.py \
  --manifest "c:/VS projects/opro2/data/processed/conditions_final/conditions_manifest_split.parquet" \
  --prompt_file results/opro_base_seed42/best_prompt.txt \
  --output_dir results/eval_base_opro \
  --batch_size 50
```

#### Etapa 7: Eval LoRA + OPRO ⭐
```bash
python scripts/evaluate_simple.py \
  --manifest "c:/VS projects/opro2/data/processed/conditions_final/conditions_manifest_split.parquet" \
  --prompt_file results/opro_lora_seed42/best_prompt.txt \
  --checkpoint checkpoints/qwen_lora_seed42/final \
  --output_dir results/eval_lora_opro \
  --batch_size 50
```

---

## Resultados Esperados

| Configuración | BA_clip | Mejora sobre Baseline | Etapa |
|---------------|---------|----------------------|-------|
| BASE (baseline) | ~80% | - | 1 |
| BASE (evaluado) | ~82% | +2% | 3a |
| LoRA (sin OPRO) | ~88% | +8% | 3b |
| **BASE + OPRO** | **~86.9%** | +6.9% | 6 |
| **LoRA + OPRO** | **~93.7%** ⭐ | **+13.7%** | 7 |

### Métricas Calculadas

- **BA_clip**: Balanced Accuracy = (Speech_Acc + NonSpeech_Acc) / 2
- **BA_conditions**: Macro-average BA sobre 4 dimensiones psicoacústicas
- **Per-condition accuracy**: 22 condiciones independientes:
  - **Duration:** 20ms, 40ms, 60ms, 80ms, 100ms, 200ms, 500ms, 1000ms
  - **SNR:** -10dB, -5dB, 0dB, 5dB, 10dB, 20dB
  - **Reverb:** none, T60=0.3s, 1.0s, 1.5s
  - **Bandpass Filter:** none, telephony, lowpass, highpass

---

## Estructura de Directorios

```
opro2_clean/
├── README.md                          # Este archivo
├── MANIFEST.md                        # Inventario completo de archivos
├── requirements.txt                   # Dependencias Python
├── config.yaml                        # Configuración global
│
├── scripts/
│   ├── evaluate_simple.py             # Evaluación principal (Etapas 1,3,6,7)
│   ├── finetune_qwen_audio.py         # LoRA training (Etapa 2)
│   ├── opro_classic_optimize.py       # OPRO con LLM local (Etapa 4,5)
│   ├── opro_post_ft_v2.py             # OPRO post-FT (Etapa 5)
│   └── run_complete_pipeline.py       # Wrapper completo
│
├── slurm/                             # Jobs de SLURM (7 etapas)
│
├── src/qsm/                           # Código fuente
│   ├── models/qwen_audio.py           # Wrapper del modelo
│   └── utils/normalize.py             # Utilidades
│
├── checkpoints/                       # Checkpoints LoRA (generados)
└── results/                           # Resultados de evaluación (generados)
```

---

## Referencias al Dataset Original

**IMPORTANTE:** Los archivos de audio NO están incluidos en este repositorio.

Todos los scripts apuntan a:
- **Dataset:** `c:/VS projects/opro2/data/`
- **Manifest:** `c:/VS projects/opro2/data/processed/conditions_final/conditions_manifest_split.parquet`
- **Audio:** `c:/VS projects/opro2/data/processed/conditions_final/{duration,snr,band,rir}/*.wav`
- **Train/Dev:** `c:/VS projects/opro2/data/processed/normalized_clips/`

---

## Detalles Técnicos

### Configuración LoRA

```yaml
lora:
  r: 64                     # Rank
  alpha: 16                 # Scaling
  dropout: 0.05
  task_type: CAUSAL_LM
  target_modules:           # 7 módulos (atención + MLP)
    - q_proj, k_proj, v_proj, o_proj
    - gate_proj, up_proj, down_proj
```

### Parámetros de Entrenamiento

- **Quantization:** 4-bit (QLoRA)
- **Batch size:** 2 × 4 gradient accumulation = 8 effective
- **Learning rate:** 5e-5
- **Epochs:** 3
- **Gradient checkpointing:** Enabled

### OPRO Optimization

- **Optimizer LLM:** Qwen/Qwen2.5-7B-Instruct (local)
- **Iterations:** 30 (BASE), 15 (LoRA)
- **Candidates per iteration:** 3
- **Top-k memory:** 10 best prompts
- **Reward function:** `R = BA_clip + 0.25 × BA_conditions - 0.05 × length_penalty`

---

## Troubleshooting

### Error: "Manifest not found"
- Verifica que el dataset original existe en `c:/VS projects/opro2/data/`
- Verifica la ruta del manifest en el mensaje de error

### Error: "CUDA out of memory"
- Reduce `--batch_size` en `evaluate_simple.py` (default: 50 → prueba 20)
- Reduce `--per_device_train_batch_size` en `finetune_qwen_audio.py` (default: 2 → prueba 1)
- Usa `export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`

### Error: "LoRA checkpoint not found"
- Ejecuta primero `01_finetune_lora.job` o `scripts/finetune_qwen_audio.py`
- Verifica que el checkpoint existe en `checkpoints/qwen_lora_seed42/final/`

### Error: "Best prompt not found"
- Ejecuta primero `02_opro_base.job` (etapa 4) o `03_opro_lora.job` (etapa 5)
- Verifica que los archivos `best_prompt.txt` existen en `results/opro_*/`

---

## Documentación Adicional

- **[MANIFEST.md](MANIFEST.md):** Inventario completo de archivos, dependencias, y comandos
- **[config.yaml](config.yaml):** Configuración global del proyecto
- **[configs/pipeline_config.yaml](configs/pipeline_config.yaml):** Configuración detallada del pipeline

---

## Contacto y Soporte

Para preguntas o problemas:
1. Revisa `MANIFEST.md` para comandos de ejemplo
2. Verifica que todas las rutas al dataset original son correctas
3. Consulta los logs en `results/` o los archivos `.out`/`.err` de SLURM

---

**Versión:** 1.0
**Última actualización:** 2025-12-11
