# Investigación: SNR-75 Thresholds Vacíos

**Fecha:** 2025-12-27
**Investigador:** Claude Code

---

## 🔍 Problema

En los resultados del análisis psicométrico, **TODOS los SNR-75 thresholds aparecen vacíos** (`{}`), incluso para los modelos de mejor performance (LoRA + OPRO Classic con BA=0.9490).

---

## ✅ Verificación de Datos

### 1. SNR Variants están en 1000ms (Correcto)

```
Verificación de archivos:
- esc50_1-45645-A-31_0152_1000ms_snr-10dB.wav
- esc50_1-45645-A-31_0152_1000ms_snr-5dB.wav
- ...

Resultado: 5,820 samples SNR, TODOS con base 1000ms ✓
```

Esto cumple con la especificación del paper: *"SNR-75: minimum SNR to reach 75% accuracy at 1000ms duration"*

### 2. Accuracy por SNR Condition (LoRA + OPRO Classic)

| SNR Condition | Accuracy | N Samples | ≥75%? |
|---------------|----------|-----------|-------|
| snr_-10dB     | 0.9742   | 970       | ✓     |
| snr_-5dB      | 0.9763   | 970       | ✓     |
| snr_0dB       | 0.9794   | 970       | ✓     |
| snr_5dB       | 0.9825   | 970       | ✓     |
| snr_10dB      | 0.9825   | 970       | ✓     |
| snr_20dB      | 0.9845   | 970       | ✓     |

**Mínima accuracy:** 97.42% (en SNR = -10dB)
**Máxima accuracy:** 98.45% (en SNR = +20dB)

---

## 💡 Conclusión

### SNR-75 está vacío porque el modelo es DEMASIADO BUENO

El threshold psicométrico SNR-75 se define como:
> *"El SNR mínimo necesario para alcanzar 75% de accuracy"*

**Problema:**
- La accuracy **nunca cruza** el umbral del 75%
- Incluso en la peor condición (SNR = -10dB), la accuracy es 97.42%
- La accuracy **siempre está por encima** del 75%

**Por lo tanto:**
- No existe un "SNR mínimo para alcanzar 75%"
- El modelo ya supera el 75% en TODAS las condiciones SNR evaluadas
- El threshold estaría **por debajo de -10dB** (fuera del rango evaluado)

---

## 🔧 Comportamiento del Código (Correcto)

El código en `statistical_analysis.py:541-545` funciona correctamente:

```python
for target in targets:
    thresh = estimate_threshold_linear(values, accuracies, target)
    if thresh is not None:
        key = f"SNR{int(target * 100)}"
        thresholds[key] = float(thresh)
    # Si thresh es None (no cruza el target), no se agrega nada
```

`estimate_threshold_linear()` verifica:
```python
if acc_df['accuracy'].min() < target_accuracy < acc_df['accuracy'].max():
    # Solo interpola si el target está DENTRO del rango
    ...
else:
    return None  # Si accuracy nunca cruza target, devuelve None
```

**Resultado:** `thresholds = {}` (diccionario vacío) ✓

---

## 📊 Comparación con Duration Thresholds (DT)

Mismo comportamiento en duraciones:

| Model             | DT50 | DT75 | DT90 | Razón                          |
|-------------------|------|------|------|--------------------------------|
| LoRA + OPRO       | -    | -    | 66ms | Ya supera 50%/75% a 20ms       |
| Base + OPRO       | -    | 37ms | 393ms| Ya supera 50% a 20ms           |
| Baseline          | -    | -    | -    | Nunca alcanza 50%/75%/90%      |

Cuando el modelo es muy bueno o muy malo:
- **Muy bueno:** Supera thresholds incluso en condiciones extremas → threshold vacío o solo DT90
- **Muy malo:** Nunca alcanza thresholds → todos vacíos

---

## ✅ Validación: Esto es CORRECTO

Los thresholds psicométricos vacíos son **información válida** que indica:

1. **Para SNR-75 vacío:**
   - El modelo es robusto al ruido
   - Mantiene >75% accuracy incluso a SNR=-10dB
   - SNR-75 < -10dB (fuera del rango evaluado)

2. **Para DT50/DT75 vacíos:**
   - El modelo tiene excelente resolución temporal
   - Ya supera 50%/75% accuracy a 20ms (mínima duración evaluada)
   - DT50/DT75 < 20ms (fuera del rango evaluado)

---

## 📝 Recomendaciones para el Paper

### Opción 1: Reportar como "No Calculable" (Recomendada)

En la tabla de resultados:
```
SNR-75: < -10dB (accuracy exceeds 75% at all evaluated SNR levels)
DT75: < 20ms (accuracy exceeds 75% at minimum duration)
```

### Opción 2: Extender el Rango de Evaluación

Para futuros experimentos:
- **SNR:** Evaluar condiciones más extremas (SNR = -15dB, -20dB)
- **Duration:** Evaluar duraciones más cortas (5ms, 10ms) si técnicamente posible

### Opción 3: Reportar la Accuracy en Condiciones Extremas

En lugar de thresholds, reportar:
```
Accuracy at SNR=-10dB: 97.42% [95% CI: ...]
Accuracy at 20ms: 91.5% [95% CI: ...]
```

---

## 📌 Resumen

**Pregunta:** ¿Por qué SNR-75 está vacío?
**Respuesta:** Porque el modelo es demasiado bueno - nunca baja del 75% de accuracy, ni siquiera a SNR=-10dB.

**Pregunta:** ¿Es esto un bug?
**Respuesta:** No, es comportamiento correcto del código de thresholds psicométricos.

**Pregunta:** ¿Qué hacer?
**Respuesta:** Reportar como "< -10dB" o "not calculable (exceeds threshold at all tested conditions)".

---

**Firmado:** Claude Code
**Verificación:** Complete ✓
