# Reporte de Verificación: Sección "Psychometric Degradation Bank"

**Fecha:** 2024-12-21
**Verificador:** Claude Code (Análisis de Codebase)
**Repositorio:** `/mnt/fast/nobackup/users/gb0048/opro2_clean`

---

## Resumen Ejecutivo

✅ **TODAS las 20 afirmaciones técnicas del paper son CORRECTAS**

- **20/20 afirmaciones verificadas** contra el código fuente
- **0 discrepancias encontradas** entre paper e implementación
- **Implementación rigurosa** con prácticas científicas adecuadas

---

## 1. General Framework

### ✅ Afirmación 1: 22 variantes en 4 dimensiones independientes

**Claim del paper:**
> "The degradation bank transforms each base clip into 22 variants along four independent dimensions."

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Archivo: [scripts/generate_experimental_variants.py:5-11](../opro2/scripts/generate_experimental_variants.py#L5-L11)

```python
# Takes base 1000ms clips and generates variants along 4 INDEPENDENT dimensions:
# 1. Duration: 20, 40, 60, 80, 100, 200, 500, 1000ms (8 values)
# 2. SNR: -10, -5, 0, +5, +10, +20 dB (6 values)
# 3. Reverb (T60): none, 0.3s, 1.0s, 2.5s (4 values)
# 4. Filter: none, bandpass, lowpass, highpass (4 values)
# Total = 22 variants per base clip (NOT 8×6×4×4 = 768 cross-product)
```

**Confirmado:** 8 + 6 + 4 + 4 = **22 condiciones independientes**

---

### ✅ Afirmación 2: Valores neutrales

**Claim del paper:**
> "When varying one dimension, the others are fixed at neutral values: 1000~ms duration, no added noise, no reverberation (T60 = 0), and no spectral filtering."

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Archivo: [scripts/generate_experimental_variants.py:61-64](../opro2/scripts/generate_experimental_variants.py#L61-L64)

```python
NEUTRAL_DURATION_MS = 1000
NEUTRAL_SNR_DB = None  # None = no noise added
NEUTRAL_T60 = 0.0  # No reverb
NEUTRAL_FILTER = 'none'
```

**Confirmado:** Exactamente los valores neutrales descritos en el paper.

---

### ✅ Afirmación 3: Container de 2000ms con padding simétrico

**Claim del paper:**
> "All variants are embedded in a 2000~ms container with symmetric padding using Gaussian noise of amplitude σ = 0.0001."

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Configuración: [scripts/generate_experimental_variants.py:67-68](../opro2/scripts/generate_experimental_variants.py#L67-L68)

```python
CONTAINER_DURATION_MS = 2000
PADDING_NOISE_AMPLITUDE = 0.0001
```

- Implementación: [src/qsm/audio/slicing.py:88-104](../opro2/src/qsm/audio/slicing.py#L88-L104)

```python
def pad_audio_to_container(audio: np.ndarray, container_duration_ms: int, sr: int,
                          noise_amplitude: float = PADDING_NOISE_AMPLITUDE) -> np.ndarray:
    """Pad audio to container duration with centered placement and low-amplitude noise."""
    # ... cálculo de padding simétrico ...

    left_noise = np.random.normal(0, noise_amplitude, left_padding).astype(np.float32)
    right_noise = np.random.normal(0, noise_amplitude, right_padding).astype(np.float32)

    return np.concatenate([left_noise, audio, right_noise])
```

**Confirmado:**
- Container: 2000ms
- Padding: Simétrico (audio centrado)
- Ruido: Gaussiano con σ = 0.0001

---

## 2. Segment Duration Conditions (§3.3.1)

### ✅ Afirmación 4: 8 duraciones específicas

**Claim del paper:**
> "We extract centered segments of 20, 40, 60, 80, 100, 200, 500, and 1000~ms from each base clip, yielding eight duration conditions."

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Configuración: [scripts/generate_experimental_variants.py:55](../opro2/scripts/generate_experimental_variants.py#L55)

```python
DEFAULT_DURATIONS_MS = [20, 40, 60, 80, 100, 200, 500, 1000]
```

- También confirmado en: [configs/pipeline_config.yaml:62](configs/pipeline_config.yaml#L62)

```yaml
durations_ms: [20, 40, 60, 80, 100, 200, 500, 1000]
```

**Confirmado:** Exactamente 8 valores de duración.

---

### ✅ Afirmación 5: Extracción centrada

**Claim del paper:**
> "We extract **centered segments** of [durations] from each base clip"

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Implementación: [src/qsm/audio/slicing.py:15-49](../opro2/src/qsm/audio/slicing.py#L15-L49)

```python
def extract_segment_center(
    audio: np.ndarray,
    duration_ms: int,
    sr: int = 16000,
) -> np.ndarray:
    """Extract a segment of specified duration from the CENTER of audio."""
    target_samples = int(duration_ms * sr / 1000.0)
    current_samples = len(audio)

    if target_samples >= current_samples:
        return audio

    # Center extraction: extract from middle of audio
    start_idx = (current_samples - target_samples) // 2
    end_idx = start_idx + target_samples

    return audio[start_idx:end_idx]
```

**Confirmado:** Extracción centrada con recorte simétrico.

---

### ✅ Afirmación 6: Sin time-stretching para clips cortos

**Claim del paper:**
> "If the source clip is shorter than the target duration, no time-stretching is applied: the full clip is kept and later padded to the 2000~ms container, rather than being artificially lengthened."

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Implementación: [src/qsm/audio/slicing.py:41-43](../opro2/src/qsm/audio/slicing.py#L41-L43)

```python
if target_samples >= current_samples:
    # If requesting same or longer duration, return entire audio
    return audio
```

**Confirmado:** Devuelve el audio completo sin estiramiento temporal.

---

## 3. SNR Manipulations (§3.3.2)

### ✅ Afirmación 7: 6 niveles de SNR específicos

**Claim del paper:**
> "We add white Gaussian noise at six SNR levels: −10, −5, 0, +5, +10, and +20~dB."

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Configuración: [scripts/generate_experimental_variants.py:56](../opro2/scripts/generate_experimental_variants.py#L56)

```python
DEFAULT_SNR_LEVELS_DB = [-10, -5, 0, 5, 10, 20]
```

- También confirmado en: [configs/pipeline_config.yaml:65](configs/pipeline_config.yaml#L65)

```yaml
snr_levels_db: [-10, -5, 0, 5, 10, 20]
```

**Confirmado:** Exactamente 6 niveles de SNR.

---

### ✅ Afirmación 8: Ruido Gaussiano blanco

**Claim del paper:**
> "We add **white Gaussian noise** at six SNR levels"

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Implementación: [src/qsm/audio/noise.py:57-59](../opro2/src/qsm/audio/noise.py#L57-L59)

```python
# Generate noise for the entire container
rng = np.random.default_rng(seed)
noise = rng.standard_normal(len(audio)).astype(audio.dtype)
```

**Confirmado:** Ruido Gaussiano blanco usando `standard_normal()` (μ=0, σ=1, luego escalado).

---

### ✅ Afirmación 9: Fórmula de calibración de ruido

**Claim del paper:**
> "Noise amplitude is calibrated to the signal's Root Mean Square (RMS):
> σ_noise = RMS_signal / 10^(SNR/20)"

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Implementación: [src/qsm/audio/noise.py:82-87](../opro2/src/qsm/audio/noise.py#L82-L87)

```python
# Scale noise to achieve target SNR
# SNR_dB = 20*log10(RMS_signal / RMS_noise)
# => RMS_noise = RMS_signal / 10^(SNR_dB/20)
target_rms_noise = rms_signal / (10 ** (snr_db / 20.0))
current_rms_noise = compute_rms(noise)
noise = noise * (target_rms_noise / current_rms_noise)
```

**Confirmado:** Fórmula exacta implementada con comentarios explicativos.

---

### ✅ Afirmación 10: Clipping a [-1, 1]

**Claim del paper:**
> "The resulting mixture is clipped to [−1, 1] to prevent saturation."

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Implementación: [scripts/generate_experimental_variants.py:118](../opro2/scripts/generate_experimental_variants.py#L118)

```python
return np.clip(noisy_audio, -1.0, 1.0)
```

**Confirmado:** Hard clipping al rango [-1.0, 1.0].

---

### ✅ Afirmación 11: Caso especial para segmentos silenciosos

**Claim del paper:**
> "For near-silent segments where the RMS of the effective speech/non-speech region falls below 10^−8, we cannot meaningfully enforce a target SNR. In these rare cases, we add a minimal amount of Gaussian noise (RMS = 10^−4), mark the SNR as undefined, and retain the sample as an extreme low-SNR condition."

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Implementación: [src/qsm/audio/noise.py:61-80](../opro2/src/qsm/audio/noise.py#L61-L80)

```python
if rms_signal < 1e-8:
    # Silent segment: use minimal noise level instead of SNR-based mixing
    import warnings

    warnings.warn(
        f"Effective segment has near-zero RMS ({rms_signal:.2e}); using minimal noise"
    )

    target_rms_noise = 1e-4  # Minimal noise level
    current_rms_noise = compute_rms(noise)
    noise = noise * (target_rms_noise / current_rms_noise)

    meta = {
        "snr_db": None,  # SNR undefined for silent segments
        "rms_signal": float(rms_signal),
        "rms_noise": float(target_rms_noise),
        "seed": seed,
        "silent_segment": True,
    }
    return audio + noise, meta
```

**Confirmado:**
- Umbral: RMS < 1×10^-8
- Ruido mínimo: RMS = 1×10^-4
- SNR marcado como `None` (undefined)

---

## 4. Reverberation Simulation (§3.3.3)

### ✅ Afirmación 12: 4 condiciones de reverberación

**Claim del paper:**
> "Four reverberation conditions are defined by target T60: 0.0~s (dry, no convolution), 0.3~s (typical of small offices or meeting rooms), 1.0~s (larger conference rooms), and 2.5~s (highly reverberant spaces such as halls or atriums)."

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Configuración: [scripts/generate_experimental_variants.py:57](../opro2/scripts/generate_experimental_variants.py#L57)

```python
DEFAULT_T60_VALUES = [0.0, 0.3, 1.0, 2.5]  # 0.0 = no reverb
```

**Confirmado:** Exactamente 4 valores de T60 como se especifica.

**Nota importante:** Los **datos reales** usan T60 = 0.0, 0.3, 1.0, **2.5s** (como dice el paper), pero el README.md de opro2_clean tiene un error y menciona 1.5s. Este error de documentación no afecta al paper.

---

### ✅ Afirmación 13: Dataset RIRS_NOISES

**Claim del paper:**
> "We simulate room acoustics using real Room Impulse Responses (RIRs) from the RIRS_NOISES dataset"

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Implementación: [src/qsm/audio/reverb.py:1-6](../opro2/src/qsm/audio/reverb.py#L1-L6)

```python
"""
Reverberation module using Room Impulse Response (RIR) convolution.

Uses OpenSLR SLR28 RIRS_NOISES dataset.
Dataset: https://www.openslr.org/28/
"""
```

- Estructura de carga: [src/qsm/audio/reverb.py:20-23](../opro2/src/qsm/audio/reverb.py#L20-L23)

```python
"""
Loads RIRs from OpenSLR SLR28 structure:
  {rir_root}/simulated_rirs/*/Room*.wav  -> simulated
  {rir_root}/real_rirs_isotropic_noises/*.wav -> real
```

**Confirmado:** Usa OpenSLR SLR28 RIRS_NOISES dataset.

---

### ✅ Afirmación 14: Tolerancia de selección de RIRs

**Claim del paper:**
> "For each non-zero T60, we select RIRs whose estimated reverberation time falls within ±0.2~s of the target."

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Selección: [scripts/generate_experimental_variants.py:238-239](../opro2/scripts/generate_experimental_variants.py#L238-L239)

```python
# Get RIRs in T60 range (±0.2s tolerance)
rir_ids = rir_database.get_by_t60(t60 - 0.2, t60 + 0.2)
```

- Método de búsqueda: [src/qsm/audio/reverb.py:104-119](../opro2/src/qsm/audio/reverb.py#L104-L119)

```python
def get_by_t60(self, t60_min: float, t60_max: float) -> list[str]:
    """Get RIR IDs within T60 range."""
    return [
        rir_id
        for rir_id, meta in self.rirs.items()
        if meta.get("T60") is not None and t60_min <= meta["T60"] <= t60_max
    ]
```

**Confirmado:** Tolerancia de ±0.2s para matching de T60.

---

### ✅ Afirmación 15: FFT convolution, truncation, RMS normalization

**Claim del paper:**
> "Convolution is performed via FFT, and the output is truncated to the original segment length... The resulting signal is normalized to match the original RMS before padding."

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Implementación: [src/qsm/audio/reverb.py:126-160](../opro2/src/qsm/audio/reverb.py#L126-L160)

```python
def apply_rir(
    audio: np.ndarray,
    rir: np.ndarray,
    normalize: bool = True,
    rir_gain: float = 1.0,
) -> np.ndarray:
    """Apply RIR to audio via convolution."""

    # Scale RIR
    rir_scaled = rir * rir_gain

    # Convolve via FFT
    reverb = signal.fftconvolve(audio, rir_scaled, mode="full")

    # Truncate to original length
    reverb = reverb[: len(audio)]

    # Normalize to preserve energy
    if normalize:
        rms_orig = np.sqrt(np.mean(audio**2))
        rms_reverb = np.sqrt(np.mean(reverb**2))
        if rms_reverb > 1e-8:
            reverb = reverb * (rms_orig / rms_reverb)

    return reverb.astype(audio.dtype)
```

**Confirmado:**
- Convolución FFT: `signal.fftconvolve()`
- Truncamiento: `reverb = reverb[: len(audio)]`
- Normalización RMS: Preserva energía de la señal original

---

## 5. Spectral Filtering (§3.3.4)

### ✅ Afirmación 16: 4 condiciones de filtrado

**Claim del paper:**
> "We apply four spectral conditions using fourth-order Butterworth filters with zero-phase filtering: none (unfiltered), bandpass (300--3400~Hz), lowpass (cutoff 3400~Hz), and highpass (cutoff 300~Hz)."

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Configuración: [scripts/generate_experimental_variants.py:58](../opro2/scripts/generate_experimental_variants.py#L58)

```python
DEFAULT_FILTER_TYPES = ['none', 'bandpass', 'lowpass', 'highpass']
```

**Confirmado:** Exactamente 4 tipos de filtro.

---

### ✅ Afirmación 17: Butterworth 4th-order, zero-phase

**Claim del paper:**
> "using **fourth-order Butterworth** filters with **zero-phase filtering**"

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Documentación: [src/qsm/audio/filters.py:1-6](../opro2/src/qsm/audio/filters.py#L1-L6)

```python
"""
Band-limited filtering module.

Implements telephony band-pass (300-3400 Hz) and ablation filters (LP/HP).
Uses zero-phase Butterworth IIR filters for clean frequency response.
"""
```

- Implementación: [src/qsm/audio/filters.py:16-64](../opro2/src/qsm/audio/filters.py#L16-L64)

```python
def apply_filter(
    audio: np.ndarray,
    sr: int,
    filter_type: FilterType,
    lowcut: float = None,
    highcut: float = None,
    order: int = 4,  # ← DEFAULT ORDER
) -> np.ndarray:
    """Apply zero-phase Butterworth filter to audio."""

    # ... diseño de filtro usando scipy.signal.butter con order=4 ...

    # Apply zero-phase filtering (forward-backward)
    filtered = signal.sosfiltfilt(sos, audio)  # ← ZERO-PHASE
    return filtered.astype(audio.dtype)
```

**Confirmado:**
- Orden: 4 (parámetro por defecto)
- Tipo: Butterworth (scipy.signal.butter)
- Zero-phase: `sosfiltfilt()` (filtrado forward-backward)

---

### ✅ Afirmación 18: Bandpass 300-3400 Hz

**Claim del paper:**
> "bandpass (300--3400~Hz)"

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Implementación: [src/qsm/audio/filters.py:67-92](../opro2/src/qsm/audio/filters.py#L67-L92)

```python
def apply_bandpass(
    audio: np.ndarray,
    sr: int,
    lowcut: float = 300.0,  # ← DEFAULT
    highcut: float = 3400.0,  # ← DEFAULT
    order: int = 4,
) -> np.ndarray:
    """
    Apply telephony band-pass filter (300-3400 Hz by default).

    Standard telephony band per ITU-T.
    """
    return apply_filter(
        audio, sr, filter_type="bandpass", lowcut=lowcut, highcut=highcut, order=order
    )
```

**Confirmado:** Bandpass 300-3400 Hz (banda telefónica ITU-T).

---

### ✅ Afirmación 19: Lowpass cutoff 3400 Hz

**Claim del paper:**
> "lowpass (cutoff 3400~Hz)"

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Implementación: [src/qsm/audio/filters.py:94-113](../opro2/src/qsm/audio/filters.py#L94-L113)

```python
def apply_lowpass(
    audio: np.ndarray,
    sr: int,
    highcut: float = 3400.0,  # ← DEFAULT
    order: int = 4,
) -> np.ndarray:
    """Apply low-pass filter (ablation: no high-pass component)."""
    return apply_filter(audio, sr, filter_type="lowpass", highcut=highcut, order=order)
```

**Confirmado:** Lowpass con cutoff en 3400 Hz.

---

### ✅ Afirmación 20: Highpass cutoff 300 Hz

**Claim del paper:**
> "highpass (cutoff 300~Hz)"

**Status:** ✅ **VERIFICADO**

**Evidencia:**
- Implementación: [src/qsm/audio/filters.py:115-134](../opro2/src/qsm/audio/filters.py#L115-L134)

```python
def apply_highpass(
    audio: np.ndarray,
    sr: int,
    lowcut: float = 300.0,  # ← DEFAULT
    order: int = 4,
) -> np.ndarray:
    """Apply high-pass filter (ablation: no low-pass component)."""
    return apply_filter(audio, sr, filter_type="highpass", lowcut=lowcut, order=order)
```

**Confirmado:** Highpass con cutoff en 300 Hz.

---

## Archivos Críticos Verificados

### Módulos de Procesamiento de Audio
- [src/qsm/audio/filters.py](../opro2/src/qsm/audio/filters.py) - Filtrado Butterworth
- [src/qsm/audio/noise.py](../opro2/src/qsm/audio/noise.py) - Manipulación de SNR
- [src/qsm/audio/reverb.py](../opro2/src/qsm/audio/reverb.py) - Convolución con RIR
- [src/qsm/audio/slicing.py](../opro2/src/qsm/audio/slicing.py) - Extracción de duración y padding

### Scripts de Generación de Variantes
- [scripts/generate_experimental_variants.py](../opro2/scripts/generate_experimental_variants.py) - Generador principal

### Archivos de Configuración
- [config.yaml](config.yaml) - Configuración global
- [configs/pipeline_config.yaml](configs/pipeline_config.yaml) - Configuración de pipeline

---

## Resumen Final

**Estado:** ✅ **TODAS las afirmaciones del paper son 100% correctas**

- ✅ **20/20 afirmaciones verificadas** contra el código fuente
- ✅ **0 discrepancias** entre paper e implementación
- ✅ **Implementación científicamente rigurosa** con prácticas adecuadas

**Valores confirmados:**

**Framework General:**
- ✅ 22 variantes independientes (8+6+4+4)
- ✅ Valores neutrales correctos (1000ms, sin ruido, T60=0, sin filtro)
- ✅ Container 2000ms con padding σ=0.0001

**Duration:**
- ✅ 8 valores: 20, 40, 60, 80, 100, 200, 500, 1000 ms
- ✅ Extracción centrada, sin time-stretching

**SNR:**
- ✅ 6 niveles: -10, -5, 0, +5, +10, +20 dB
- ✅ Ruido Gaussiano blanco
- ✅ Fórmula: σ_noise = RMS_signal / 10^(SNR/20)
- ✅ Clipping [-1, 1]
- ✅ Caso especial RMS < 10^-8

**Reverberación:**
- ✅ 4 valores T60: 0.0, 0.3, 1.0, **2.5s**
- ✅ Dataset RIRS_NOISES (OpenSLR SLR28)
- ✅ Tolerancia ±0.2s
- ✅ FFT convolution, truncado, normalización RMS

**Filtrado:**
- ✅ 4 tipos: none, bandpass, lowpass, highpass
- ✅ Butterworth 4th-order, zero-phase
- ✅ Bandpass: 300-3400 Hz (ITU-T)
- ✅ Lowpass: 3400 Hz
- ✅ Highpass: 300 Hz

**Conclusión:** La sección "Psychometric Degradation Bank" del paper refleja fielmente la implementación. **No se requieren correcciones al paper.**

---

## Nota sobre Valores de Reverberación

El paper menciona correctamente **T60 = 2.5s** para la condición más reverberante, lo cual coincide con los **datos reales** verificados en:
- `experimental_variants_large/train_metadata.csv`: T60 = 0.3, 1.0, 2.5
- `variants_validated_1000/dev_metadata.csv`: T60 = 0.3, 1.0, 2.5

Sin embargo, el [README.md:229](README.md#L229) de `opro2_clean` contiene un **error de documentación** y menciona 1.5s en lugar de 2.5s. Este error no afecta al paper ni a los resultados, es solo un error tipográfico en la documentación interna.
