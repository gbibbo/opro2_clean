# Corrección Importante: Soporte Multi-Formato de Prompts

**Fecha:** 2024-12-21
**Tipo:** Corrección de verificación
**Afecta a:** VERIFICATION_REPORT_DATASET.md, resúmenes anteriores

---

## Corrección Necesaria

En verificaciones anteriores, afirmé que el sistema usa **"prompts binarios"** exclusivamente. Esto es **INCORRECTO**.

## Implementación Real

El sistema implementa **soporte completo para múltiples formatos de prompts**, incluyendo:

### 1. Formatos Soportados

**[normalize.py:36](src/qsm/utils/normalize.py#L36)**:
```python
mode: Format mode ("ab", "mc", "labels", "open", "auto")
```

**Tipos de prompts:**
1. **Binary (A/B)**: "Choose A) SPEECH or B) NONSPEECH"
2. **Labels**: "Answer SPEECH or NONSPEECH"
3. **Multiple Choice (A/B/C/D)**: 4 opciones
4. **Open-ended**: "What do you hear in this audio?"

### 2. Modificación Clave para Prompts Open-Ended

**[opro_classic_optimize.py:230-235](scripts/opro_classic_optimize.py#L230-L235)**:
```python
# REMOVED: Keyword restriction to allow open-ended prompts
# The normalize_to_binary() function handles various response formats including:
# - Binary labels (SPEECH/NONSPEECH)
# - Yes/No responses
# - Synonyms (voice, talking, music, noise, etc.)
# - Open descriptions
```

**[opro_classic_optimize.py:395](scripts/opro_classic_optimize.py#L395)**:
```python
# CONSTRAINTS:
# - Prompts can be ANY format: questions, commands, statements, binary choice, open-ended
# - The model's response will be automatically parsed to determine speech detection
```

### 3. Archivo de Pruebas Dedicado

**[test_open_prompts.py](test_open_prompts.py)** - Suite completa de pruebas para prompts open-ended:

```python
# Línea 263
"Testing modifications to allow open-ended prompts in OPRO"

# Ejemplos de prompts open-ended probados (líneas 77-81):
test_prompts = [
    ("What do you hear in this audio?", True),
    ("Describe the sound.", True),
    ("What type of audio is this?", True),
    ("Is this SPEECH or NONSPEECH?", True),  # También soporta binarios
]
```

**Ejemplos de respuestas open-ended normalizadas correctamente (líneas 127-139)**:
```python
# SPEECH
("I hear a person talking", "SPEECH", "Open: person talking"),
("This is a human voice speaking", "SPEECH", "Open: human voice"),
("Someone is having a conversation", "SPEECH", "Open: conversation"),

# NONSPEECH
("This is music", "NONSPEECH", "Open: music"),
("I hear background noise", "NONSPEECH", "Open: noise"),
("There's silence", "NONSPEECH", "Open: silence"),
```

### 4. Normalización Multi-Formato

**[normalize.py:13-183](src/qsm/utils/normalize.py#L13-L183)** - Sistema de prioridades:

1. **Priority 1**: NONSPEECH/NON-SPEECH keywords
2. **Priority 2**: SPEECH keyword
3. **Priority 3**: Letter mapping (A/B/C/D)
4. **Priority 4**: Yes/No responses
5. **Priority 5**: **Synonyms** (voice, talking, music, noise, silence, etc.)
6. **Priority 6**: LLM fallback para casos ambiguos

**El sistema puede normalizar:**
- "A" → SPEECH (si mapping = {"A": "SPEECH"})
- "SPEECH" → SPEECH
- "I hear a person talking" → SPEECH (vía synonyms)
- "This is music" → NONSPEECH (vía synonyms)
- "Background music with someone speaking" → SPEECH (vía LLM fallback)

---

## Implicaciones para el Paper

### ✅ Lo que el paper DEBE decir:

El paper debe aclarar que el sistema:

1. **Prompt por defecto**: Usa formato binario A/B como baseline
   ```
   "Choose one:
   A) SPEECH (human voice)
   B) NONSPEECH (music/noise/silence/animals)
   Answer with A or B ONLY."
   ```

2. **OPRO puede generar**: Prompts en **cualquier formato** (binarios, labels, open-ended)
   - El meta-prompt de OPRO instruye al LLM: "Prompts can be ANY format: questions, commands, statements, binary choice, open-ended"

3. **Normalización**: Robusto sistema multi-formato que convierte respuestas de cualquier tipo a SPEECH/NONSPEECH
   - Soporte explícito para respuestas open-ended vía synonym matching y LLM fallback

### ❌ Lo que el paper NO debe decir:

- ❌ "El sistema usa **exclusivamente** prompts binarios"
- ❌ "Solo se permiten prompts que contengan las palabras SPEECH/NONSPEECH"
- ❌ "Las respuestas deben ser A/B"

### ✅ Formulación correcta para el paper:

> "While the baseline prompt uses a binary A/B format, the OPRO optimization allows exploration of diverse prompt formats including open-ended questions. The model's textual response is normalized to binary SPEECH/NONSPEECH labels using a multi-format parser that handles direct labels, letter choices, yes/no responses, semantic descriptions, and ambiguous cases via LLM fallback."

O más conciso:

> "The model's textual response is normalized to binary SPEECH/NONSPEECH labels using a rule-based parser with LLM fallback, supporting multiple prompt formats including binary choice (A/B), direct labels, and open-ended questions."

---

## Archivos Críticos de Evidencia

1. **[test_open_prompts.py](test_open_prompts.py)** - Suite de pruebas completa para open-ended
2. **[src/qsm/utils/normalize.py](src/qsm/utils/normalize.py)** - Normalizador multi-formato
3. **[scripts/opro_classic_optimize.py](scripts/opro_classic_optimize.py)** - OPRO sin restricción de keywords
4. **[src/qsm/models/qwen_audio.py](src/qsm/models/qwen_audio.py)** - Modelo con prompt por defecto A/B

---

## Acción Requerida

**Revisar el paper** para asegurar que:

1. ✅ No afirma que solo se usan prompts binarios
2. ✅ Menciona el soporte multi-formato si es relevante para los resultados
3. ✅ Describe correctamente el sistema de normalización

**Si el paper menciona prompts**, usar formulación como:
- "prompts optimizados mediante OPRO" (sin especificar formato)
- "diversos formatos de prompts incluyendo open-ended"
- "sistema de normalización robusto para múltiples formatos"

---

## Resumen

- ❌ **INCORRECTO**: "El sistema usa prompts binarios"
- ✅ **CORRECTO**: "El sistema soporta múltiples formatos de prompts (binary, labels, open-ended) con normalización automática a SPEECH/NONSPEECH"

La implementación es **más flexible y robusta** de lo que indicaba mi verificación inicial.
