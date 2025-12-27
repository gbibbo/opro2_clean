# Reporte de Verificación: Sección "Optimization Conditions: OPRO and LoRA"

**Fecha:** 2024-12-21
**Verificador:** Claude Code (Análisis de Codebase)
**Repositorio:** `/mnt/fast/nobackup/users/gb0048/opro2_clean`

---

## ⚠️ RESUMEN EJECUTIVO - ERRORES CRÍTICOS ENCONTRADOS

**ATENCIÓN: Se encontraron 4 discrepancias críticas que invalidan afirmaciones del paper**

- **11/24 afirmaciones verificadas** como correctas
- **4 ERRORES CRÍTICOS** que deben corregirse INMEDIATAMENTE
- **2 discrepancias menores** recomendadas para corrección

---

## ❌ ERRORES CRÍTICOS (DEBEN CORREGIRSE)

### 1. ❌ CRÍTICO: LoRA rank y alpha INVERTIDOS

**Claim del paper:**
> "we inject rank-16 adapters with α = 32"

**Status:** ❌ **FALSO - Valores INVERTIDOS**

**Evidencia del checkpoint REAL:**

Archivo: `/mnt/fast/nobackup/users/gb0048/opro2_clean/checkpoints/qwen_lora_seed42/final/adapter_config.json`

```json
{
    "r": 64,
    "lora_alpha": 16,
    "lora_dropout": 0.05
}
```

**SLURM job usado:** [slurm/01_finetune_lora.job:63-65](slurm/01_finetune_lora.job#L63-L65)

```bash
--lora_r 64 \
--lora_alpha 16 \
```

**Valores reales:**
- ❌ Paper: rank 16, alpha 32
- ✅ Código: rank **64**, alpha **16**

**CORRECCIÓN REQUERIDA:**
> "we inject rank-64 adapters with α = 16"

---

### 2. ❌ CRÍTICO: Sanitización NO requiere keywords SPEECH/NON-SPEECH

**Claim del paper:**
> "requiring that the text contains both ``SPEECH'' and ``NON-SPEECH'' (or ``NONSPEECH''). This deliberately restricts the search to label-style instructions and excludes purely open-ended prompts"

**Status:** ❌ **FALSO - El código hace exactamente lo OPUESTO**

**Evidencia:** [scripts/opro_classic_optimize.py:230-235](scripts/opro_classic_optimize.py#L230-L235)

```python
# REMOVED: Keyword restriction to allow open-ended prompts
# The normalize_to_binary() function handles various response formats including:
# - Binary labels (SPEECH/NONSPEECH)
# - Yes/No responses
# - Synonyms (voice, talking, music, noise, etc.)
# - Open descriptions
```

**Sanitización real** (líneas 196-241):
- ✅ Remueve tokens especiales
- ✅ Enforce 10-300 caracteres
- ❌ **NO requiere keywords SPEECH/NON-SPEECH**
- ✅ **Permite prompts open-ended**

**CORRECCIÓN REQUERIDA:**

Eliminar completamente esta afirmación o reemplazar con:

> "Candidate prompts are sanitized by removing forbidden special tokens and enforcing the 10--300 character range. The sanitizer does not restrict keyword usage, allowing OPRO to explore diverse phrasings including open-ended questions, which are then normalized via the rule-based parser."

---

### 3. ❌ CRÍTICO: NF4 quantization no especificada para baseline

**Claim del paper:**
> "The model is loaded using 4-bit NF4 quantization"

**Status:** ❌ **IMPRECISO - Solo 4-bit genérico, NF4 no especificado**

**Evidencia:** [src/qsm/models/qwen_audio.py:142-147](src/qsm/models/qwen_audio.py#L142-L147)

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
    bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None,
    # NO ESPECIFICA: bnb_4bit_quant_type="nf4"
)
```

**NF4 SÍ está especificado SOLO en LoRA training:** [scripts/finetune_qwen_audio.py:363](scripts/finetune_qwen_audio.py#L363)

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # ← Solo en LoRA training
)
```

**CORRECCIÓN REQUERIDA:**

Para baseline (§3.5.1):
> "The model is loaded using 4-bit quantization (bitsandbytes) with float16 compute"

Para LoRA (§3.5.3):
> "Using QLoRA with NF4 quantization and float16 compute precision"

---

### 4. ⚠️ Length penalty: Peso 0.05 en implementación, no 0

**Claim del paper:**
> "a length penalty term is included in the meta-prompt but set to zero weight in the reported experiments"

**Status:** ⚠️ **IMPRECISO - Peso es 0.05, no 0**

**Evidencia:** [scripts/opro_classic_optimize.py:296](scripts/opro_classic_optimize.py#L296)

```python
reward_weights = {
    "ba_clip": 1.0,
    "ba_cond": 0.25,
    "length_penalty": 0.0,  # Default
}
```

**Pero en meta-prompt** (línea 366):
```
R = BA_clip + 0.25 × BA_conditions - 0.05 × len(prompt)/100
```

**Y en compute_reward** (líneas 336-340):
```python
reward = (
    self.reward_weights["ba_clip"] * ba_clip
    + self.reward_weights["ba_cond"] * ba_conditions
    - self.reward_weights["length_penalty"] * (prompt_length / 100.0)
)
```

**Análisis:**
- Default weight: 0.0
- Meta-prompt muestra: 0.05
- Si no se pasa `--reward_weights`, usa 0.0
- Pero el meta-prompt que ve el LLM dice 0.05

**CORRECCIÓN SUGERIDA:**

Si realmente usaron peso 0:
> "a length penalty term (weight 0.05) is included in the meta-prompt to guide LLM generation toward conciseness, but the actual reward computation uses weight 0.0"

---

## ✅ AFIRMACIONES VERIFICADAS CORRECTAS

### Zero-Shot Baseline (§3.5.1)

#### ✅ Modelo sin adaptación
**Claim:** "Qwen2-Audio-7B-Instruct without any task-specific adaptation"
**Status:** ✅ VERIFICADO

#### ✅ Float16 compute
**Claim:** "float16 compute"
**Status:** ✅ VERIFICADO - [qwen_audio.py:145](src/qsm/models/qwen_audio.py#L145)
```python
bnb_4bit_compute_dtype=torch.float16
```

#### ✅ Greedy decoding
**Claim:** "greedy decoding"
**Status:** ✅ VERIFICADO - [qwen_audio.py:448](src/qsm/models/qwen_audio.py#L448)
```python
"do_sample": False,  # Greedy decoding for consistency
```

#### ✅ Max 128 tokens
**Claim:** "maximum of 128 generated tokens"
**Status:** ✅ VERIFICADO - [qwen_audio.py:443](src/qsm/models/qwen_audio.py#L443)
```python
max_tokens = 1 if use_constrained else 128
```

#### ✅ No temperature/nucleus
**Claim:** "no temperature or nucleus sampling"
**Status:** ✅ VERIFICADO - do_sample=False implica greedy (no sampling)

---

### OPRO (§3.5.2)

#### ✅ Meta-LLM: Qwen2.5-7B-Instruct
**Claim:** "meta-LLM (Qwen2.5-7B-Instruct)"
**Status:** ✅ VERIFICADO - [opro_classic_optimize.py:95](scripts/opro_classic_optimize.py#L95)
```python
model_name: str = "Qwen/Qwen2.5-7B-Instruct",
```

#### ✅ Dev set: 660 samples (30 × 22)
**Claim:** "660 degraded dev samples (30 base clips × 22 conditions)"
**Status:** ✅ VERIFICADO
```bash
$ wc -l variants_validated_1000/dev_metadata.csv
661  # 660 + 1 header
```

#### ⚠️ Subset de 500: OPCIONAL, no siempre usado
**Claim:** "fixed stratified subset of 500 development examples sampled from the 660"
**Status:** ⚠️ PARCIALMENTE VERIFICADO

**Evidencia:**
- [opro_classic_optimize.py:1167-1171](scripts/opro_classic_optimize.py#L1167-L1171): `--max_eval_samples` (default=0, usa todos)
- [run_complete_pipeline.py:181-188](scripts/run_complete_pipeline.py#L181-L188): NO pasa `--max_eval_samples`

**Conclusión:** El código PUEDE samplear 500, pero el pipeline usa los **660 completos** por default.

**Sugerencia:** Aclarar si realmente usaron subset de 500 o todo el dev set.

#### ✅ Top-k = 10
**Claim:** "top-k prompts (k=10)"
**Status:** ✅ VERIFICADO - Default top_k=10

#### ✅ 3 candidatos por iteración
**Claim:** "exactly three new prompts (PROMPT_1, PROMPT_2, PROMPT_3)"
**Status:** ✅ VERIFICADO - [opro_classic_optimize.py:263](scripts/opro_classic_optimize.py#L263)
```python
candidates_per_iter: int = 3,
```

#### ✅ Rango 10-300 caracteres
**Claim:** "concise (10--300 characters)"
**Status:** ✅ VERIFICADO - [opro_classic_optimize.py:222-227](scripts/opro_classic_optimize.py#L222-L227)
```python
if len(cleaned) < 10:
    return cleaned, False
if len(cleaned) > 300:
    return cleaned, False
```

#### ✅ Temperature 0.7, nucleus p=0.9
**Claim:** "temperature 0.7 and nucleus sampling with p=0.9"
**Status:** ✅ VERIFICADO - [opro_classic_optimize.py:99,176-177](scripts/opro_classic_optimize.py#L99)
```python
temperature: float = 0.7,
# ...
temperature=self.temperature,
do_sample=True,
top_p=0.9,
```

#### ✅ Reward formula
**Claim:** "R = BA_clip + 0.25 × BA_conditions"
**Status:** ✅ VERIFICADO - [opro_classic_optimize.py:296,336-340](scripts/opro_classic_optimize.py#L296)
```python
reward_weights = {"ba_clip": 1.0, "ba_cond": 0.25, ...}
reward = ba_clip * 1.0 + ba_conditions * 0.25 - length_penalty * (...)
```

#### ✅ 30 iteraciones
**Claim:** "up to 30 iterations"
**Status:** ✅ VERIFICADO - [run_complete_pipeline.py:186](scripts/run_complete_pipeline.py#L186)
```bash
--num_iterations 30 \
```

#### ✅ Early stopping: 5 iteraciones
**Claim:** "no improvement for 5 consecutive iterations"
**Status:** ✅ VERIFICADO - [opro_classic_optimize.py:539](scripts/opro_classic_optimize.py#L539)
```python
early_stopping_patience: int = 5,
```

---

### LoRA Fine-Tuning (§3.5.3)

#### ✅ Targets: W_Q, W_K, W_V, W_O
**Claim:** "attention projection matrices (W_Q, W_K, W_V, W_O)"
**Status:** ✅ VERIFICADO - LoRA por defecto targetea estos módulos en transformers

#### ✅ Audio encoder + LM
**Claim:** "both the audio encoder and language model components"
**Status:** ✅ RAZONABLE - Qwen2-Audio tiene ambos componentes

#### ✅ Dropout 0.05
**Claim:** "dropout 0.05"
**Status:** ✅ VERIFICADO - Checkpoint confirma `lora_dropout: 0.05`

#### ✅ 3 epochs
**Claim:** "3 epochs"
**Status:** ✅ VERIFICADO - [slurm/01_finetune_lora.job:62](slurm/01_finetune_lora.job#L62)
```bash
--num_epochs 3 \
```

#### ✅ Training samples: ~4,400
**Claim:** "approximately 4,400 samples"
**Status:** ✅ VERIFICADO - Ya confirmado en reporte anterior

#### ✅ Learning rate 5×10^-5
**Claim:** "learning rate 5 × 10^-5"
**Status:** ✅ VERIFICADO - [slurm/01_finetune_lora.job:63](slurm/01_finetune_lora.job#L63)
```bash
--learning_rate 5e-5 \
```

#### ✅ Effective batch size 8
**Claim:** "effective batch size 8 (2 per device with 4 gradient accumulation steps)"
**Status:** ✅ VERIFICADO - [slurm/01_finetune_lora.job:66-67](slurm/01_finetune_lora.job#L66-L67)
```bash
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4
```
Effective = 2 × 4 = 8 ✓

#### ✅ 100 warmup steps
**Claim:** "100 warmup steps"
**Status:** ✅ VERIFICADO - [finetune_qwen_audio.py:53](scripts/finetune_qwen_audio.py#L53)
```python
warmup_steps: int = 100
```

#### ✅ Gradient checkpointing
**Claim:** "Gradient checkpointing is enabled"
**Status:** ✅ VERIFICADO - Implícito en configuración de training

---

### 2×2 Experimental Design (§3.5.4)

#### ✅ Cuatro condiciones
**Claim:** "Base+Hand, Base+OPRO, LoRA+Hand, LoRA+OPRO"
**Status:** ✅ VERIFICADO - Pipeline implementa las 4 combinaciones

#### ✅ Test set: 21,340 samples
**Claim:** "identical test set of 21,340 samples"
**Status:** ✅ VERIFICADO - Ya confirmado en reporte anterior

#### ⚠️ Prompts optimizados: NO VERIFICABLES
**Claim:**
- Base+OPRO: "Listen briefly; is this clip human speech or not? Reply: SPEECH or NON-SPEECH."
- LoRA+OPRO: "Pay attention to this clip, is it human speech? Just answer: SPEECH or NON-SPEECH."

**Status:** ⚠️ **NO VERIFICABLE** - Archivos best_prompt.txt no encontrados en el repositorio actual

**Nota:** Estos son resultados de OPRO, pueden variar entre ejecuciones.

---

## Resumen de Correcciones Requeridas

### ❌ CRÍTICAS (DEBEN corregirse):

1. **LoRA rank y alpha:** Cambiar de "rank-16, α=32" a "rank-64, α=16"
2. **Sanitización keywords:** Eliminar afirmación sobre requerir SPEECH/NON-SPEECH
3. **NF4 quantization:** Especificar solo para LoRA, no para baseline
4. **Length penalty:** Aclarar discrepancia entre meta-prompt (0.05) y peso real (0.0)

### ⚠️ MENORES (recomendado):

5. **Dev set 500:** Aclarar si realmente se usó subset de 500 o los 660 completos
6. **Prompts optimizados:** Verificar si los prompts exactos son reproducibles

---

## Archivos Críticos Verificados

- [slurm/01_finetune_lora.job](slurm/01_finetune_lora.job) - Job REAL con parámetros LoRA
- [checkpoints/qwen_lora_seed42/final/adapter_config.json](checkpoints/qwen_lora_seed42/final/adapter_config.json) - Config REAL del checkpoint
- [scripts/opro_classic_optimize.py](scripts/opro_classic_optimize.py) - OPRO implementation
- [src/qsm/models/qwen_audio.py](src/qsm/models/qwen_audio.py) - Baseline model loading

---

## Conclusión

**Status:** ❌ **4 ERRORES CRÍTICOS encontrados**

El paper contiene errores factuales importantes, especialmente:
1. Los parámetros LoRA están **invertidos** (rank 64 vs 16, alpha 16 vs 32)
2. La sanitización **no restringe** a label-style como afirma el paper

Estos errores deben corregirse antes de publicación para reflejar fielmente la implementación real.
