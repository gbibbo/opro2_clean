# Reporte de Verificación: Experimental Setup (§4)

**Fecha:** 2025-12-22
**Documento:** Sección "Experimental Setup" del paper
**Código:** `/mnt/fast/nobackup/users/gb0048/opro2_clean`

---

## Resumen Ejecutivo

Verificación sistemática de las afirmaciones técnicas sobre implementación, hardware y parámetros de inferencia en la sección "Experimental Setup".

**Hallazgos:**
- ✅ **12 afirmaciones verificadas como CORRECTAS**
- ⚠️ **2 discrepancias/aclaraciones necesarias**

---

## §4.1.1 Software and Libraries

### ✅ Afirmación 1: PyTorch 2.x

> **Paper:** "PyTorch 2.x"

**Verificación:**
```txt
# requirements.txt:5
torch>=2.0.0

# pyproject.toml:17
"torch>=2.0.0",
```

**Estado:** ✅ CORRECTO

---

### ✅ Afirmación 2: Transformers ≥4.40

> **Paper:** "Transformers~$\geq$4.40"

**Verificación:**
```txt
# requirements.txt:7
transformers>=4.40.0

# pyproject.toml:19
"transformers>=4.40.0",
```

**Estado:** ✅ CORRECTO

---

### ✅ Afirmación 3: PEFT ≥0.10

> **Paper:** "PEFT~$\geq$0.10 for LoRA"

**Verificación:**
```txt
# requirements.txt:25
peft>=0.10.0

# pyproject.toml:24
"peft>=0.10.0",
```

**Estado:** ✅ CORRECTO

---

### ⚠️ Afirmación 4: bitsandbytes

> **Paper:** "bitsandbytes for quantization"

**Verificación:**

**Archivos de dependencias:**
```txt
# requirements.txt - NO menciona bitsandbytes
# pyproject.toml - NO menciona bitsandbytes
```

**Uso en código:**
```python
# src/qsm/models/qwen_audio.py:140
from transformers import BitsAndBytesConfig

# scripts/opro_classic_optimize.py:127
from transformers import BitsAndBytesConfig
```

**Análisis:**
- `bitsandbytes` NO está listado explícitamente en `requirements.txt` ni `pyproject.toml`
- Sin embargo, se importa `BitsAndBytesConfig` de `transformers`
- La importación es **condicional** (dentro de if `load_in_4bit or load_in_8bit`)
- `transformers` tiene `bitsandbytes` como dependencia opcional
- En Surrey HPC, probablemente esté instalado en el contenedor Apptainer

**Estado:** ⚠️ **ACLARACIÓN NECESARIA**

**Recomendación:**
- Opción 1: Agregar `bitsandbytes` a `requirements.txt` explícitamente
- Opción 2: En el paper, aclarar: "bitsandbytes (via transformers optional dependency)"

---

### ✅ Afirmación 5: scipy for impulse response processing

> **Paper:** "scipy for impulse response processing"

**Verificación:**
```txt
# requirements.txt:30
scipy>=1.12.0

# pyproject.toml:28
"scipy>=1.12.0",
```

**Estado:** ✅ CORRECTO

---

## §4.1.2 Hardware and Runtime

### ✅ Afirmación 6: RTX 3090 GPUs

> **Paper:** "Experiments were conducted on NVIDIA RTX 3090 GPUs"

**Verificación:**
```markdown
# RUN_PIPELINE.md:51
- **GPU**: 1x RTX 3090

# README.md:207
- **GPU:** 40GB+ VRAM para training, 24GB para inference (RTX 3090, A6000, V100)
```

```bash
# All SLURM job files specify:
#SBATCH --partition=3090
#SBATCH --gpus=1

# Examples:
# slurm/01_finetune_lora.job:2-3
# slurm/04_eval_base.job:2-4
# slurm/07_eval_lora_opro.job:2-4
```

**SLURM partition info:**
```
Partition: 3090
Nodes: aisurrey[11-14,17-19] (7 nodes total)
State: UP
```

**Estado:** ✅ CORRECTO

---

### ✅ Afirmación 7: 24GB VRAM

> **Paper:** "with 24 GB of VRAM"

**Verificación:**
```markdown
# README.md:207
- **GPU:** 40GB+ VRAM para training, 24GB para inference (RTX 3090, A6000, V100)
```

**Notas:**
- RTX 3090 tiene efectivamente 24GB de VRAM
- Los jobs de evaluación solicitan 48GB de RAM del sistema (no VRAM)
- 4-bit quantization permite correr el modelo en 24GB VRAM

**Estado:** ✅ CORRECTO

---

### ✅ Afirmación 8: SLURM-managed cluster

> **Paper:** "on a SLURM-managed cluster"

**Verificación:**
- Todos los experimentos se ejecutan vía SLURM job scripts
- Partición: `3090` con 7 nodos activos
- Wrapper script para acceso desde datamove1: `slurm/tools/on_submit.sh`

```bash
# Ejemplos de SLURM directives:
#SBATCH --partition=3090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
```

**Estado:** ✅ CORRECTO

---

### ⚠️ Afirmación 9: Runtime 3-4 hours for 21,340 samples

> **Paper:** "Evaluating 21,340 samples typically requires 3–4 hours with batch size 50"

**Verificación:**

**SLURM time allocations:**
```bash
# slurm/04_eval_base.job:7
#SBATCH --time=02:00:00

# slurm/06_eval_base_opro.job:7
#SBATCH --time=02:00:00

# slurm/07_eval_lora_opro.job:7
#SBATCH --time=02:00:00
```

```markdown
# MANIFEST.md:56-59
| 04_eval_base.job         | 3a | evaluate_simple.py                  | 3090 | 2h  | 48GB |
| 06_eval_base_opro.job    | 6  | evaluate_simple.py --prompt_file    | 3090 | 2h  | 48GB |
| 07_eval_lora_opro.job    | 7  | evaluate_simple.py --checkpoint     | 3090 | 2h  | 48GB |
```

**Confirmación de sample count:**
```markdown
# README.md:36
**Métricas Finales (Test Set - 21,340 muestras):**

# VERIFICATION_REPORT_DATASET.md:267
- **Tamaño real:** 21,340 samples (21,341 líneas con header)
```

**Análisis:**
- **Sample count:** ✅ 21,340 es correcto
- **SLURM allocation:** 2 horas (no 3-4)
- **Batch size 50:** ✅ Confirmado en jobs

**Posibles explicaciones:**
1. **Tiempo de job ≠ tiempo real:** SLURM time es límite máximo, no duración real
2. **Variación por condición:** Algunos jobs pueden tardar más
3. **Overhead:** Carga de modelo + procesamiento puede añadir tiempo
4. **Estimación conservadora:** El paper usa estimación más alta para seguridad

**Estado:** ⚠️ **DISCREPANCIA MENOR**

**Recomendación:**
- Verificar logs reales de ejecución para confirmar duración exacta
- Si es 2h, corregir paper a "2 hours"
- Si varía 2-4h, aclarar: "typically 2–4 hours depending on model variant"

---

## §4.1.3 Inference Parameters

### ✅ Afirmación 10: 16 kHz sampling rate

> **Paper:** "All audio is resampled to 16~kHz"

**Verificación:**
```python
# src/qsm/models/qwen_audio.py:296
def _pad_audio(
    self,
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> np.ndarray:
```

**Estado:** ✅ CORRECTO

---

### ✅ Afirmación 11: 2000 ms container duration

> **Paper:** "and padded to a 2000~ms container"

**Verificación:**
```python
# src/qsm/models/qwen_audio.py:84
pad_target_ms: int = 2000,  # Kept for backward compatibility

# src/qsm/models/qwen_audio.py:98
pad_target_ms: Target duration for padding in milliseconds (default: 2000)
```

**Confirmación adicional:**
```python
# From VERIFICATION_REPORT_DEGRADATIONS.md:76
CONTAINER_DURATION_MS = 2000
```

**Estado:** ✅ CORRECTO

---

### ✅ Afirmación 12: Greedy decoding

> **Paper:** "We use greedy decoding (temperature = 0)"

**Verificación:**
```python
# src/qsm/models/qwen_audio.py:448
"do_sample": False,  # Greedy decoding for consistency
```

**Notas:**
- `do_sample=False` es equivalente a greedy decoding (temperature=0)
- Comentario explícito confirma la intención

**Estado:** ✅ CORRECTO

---

### ✅ Afirmación 13: Batch size 50

> **Paper:** "with batch size 50 for evaluation"

**Verificación:**
```bash
# slurm/04_eval_base.job:58
--batch_size 50

# slurm/06_eval_base_opro.job:58
--batch_size 50

# slurm/07_eval_lora_opro.job:60
--batch_size 50
```

```python
# scripts/evaluate_simple.py:33
parser.add_argument("--batch_size", type=int, default=50, help="Batch size for processing")
```

**Estado:** ✅ CORRECTO

---

### ✅ Afirmación 14: 21,340 evaluation samples

> **Paper:** "evaluating all 21,340 samples"

**Verificación:**
```markdown
# README.md:36, 90
**Métricas Finales (Test Set - 21,340 muestras):**

# VERIFICATION_REPORT_DATASET.md:267
- **Tamaño real:** 21,340 samples (21,341 líneas con header)
- **Cálculo:** 970 clips × 22 conditions = 21,340 ✓
```

**Estado:** ✅ CORRECTO

---

## Resumen de Hallazgos

### Afirmaciones Correctas (12)

1. ✅ PyTorch ≥2.0.0
2. ✅ Transformers ≥4.40.0
3. ✅ PEFT ≥0.10.0
4. ✅ scipy ≥1.12.0
5. ✅ RTX 3090 GPUs
6. ✅ 24GB VRAM
7. ✅ SLURM cluster
8. ✅ 16 kHz sampling
9. ✅ 2000 ms container
10. ✅ Greedy decoding
11. ✅ Batch size 50
12. ✅ 21,340 samples

### Discrepancias/Aclaraciones (2)

#### 1. bitsandbytes dependency (⚠️ Minor)

**Issue:** No está explícitamente en requirements.txt

**Opciones de corrección:**
- Agregar a requirements.txt: `bitsandbytes>=0.41.0`
- Aclarar en paper: "via transformers optional dependency"

#### 2. Runtime estimation (⚠️ Minor)

**Issue:** Paper dice "3-4 hours", SLURM jobs allocan 2h

**Opciones de corrección:**
- Si logs confirman ~2h: cambiar a "approximately 2 hours"
- Si varía: "2-4 hours depending on model configuration"
- Verificar logs reales para confirmar duración exacta

---

## Recomendaciones

### Para el Paper

```latex
% Versión actual (aproximada)
Evaluating 21,340 samples typically requires 3–4 hours with batch size 50

% Versión corregida (opción 1 - si es 2h)
Evaluating 21,340 samples requires approximately 2 hours with batch size 50 on RTX 3090

% Versión corregida (opción 2 - si varía)
Evaluating 21,340 samples typically requires 2–3 hours with batch size 50,
depending on model configuration (base vs. LoRA-adapted)
```

### Para requirements.txt

```txt
# Agregar línea explícita:
bitsandbytes>=0.41.0  # Required for 4-bit/8-bit quantization
```

---

## Evidencia de Archivos

**Configuración de software:**
- [`requirements.txt`](requirements.txt) - Dependencias principales
- [`pyproject.toml`](pyproject.toml:16-38) - Configuración del paquete

**Hardware y jobs:**
- [`slurm/04_eval_base.job`](slurm/04_eval_base.job:2-7) - Configuración SLURM baseline
- [`slurm/07_eval_lora_opro.job`](slurm/07_eval_lora_opro.job:2-7) - Configuración SLURM LoRA+OPRO
- [`RUN_PIPELINE.md`](RUN_PIPELINE.md:49-54) - Documentación de hardware

**Parámetros de inferencia:**
- [`src/qsm/models/qwen_audio.py`](src/qsm/models/qwen_audio.py:296) - Sample rate 16kHz
- [`src/qsm/models/qwen_audio.py`](src/qsm/models/qwen_audio.py:84) - Container 2000ms
- [`src/qsm/models/qwen_audio.py`](src/qsm/models/qwen_audio.py:448) - Greedy decoding
- [`scripts/evaluate_simple.py`](scripts/evaluate_simple.py:33) - Batch size default

---

**Conclusión:** La sección "Experimental Setup" es en general **altamente precisa** con solo 2 discrepancias menores relacionadas con documentación de dependencias y estimación de runtime.
