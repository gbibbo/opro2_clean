# Figure 1: Degradation Bank and Experimental Design

**Generated:** 2026-01-06
**Script:** `scripts/generate_figure1_degradation_bank.py`

## Overview

This figure illustrates the psychometric degradation bank and the 2×2 experimental design used in the OPRO study.

## Panel Structure

### Panel (a): Pipeline Schematic
Shows the complete processing pipeline:
1. Base clip (1000 ms) with SPEECH/NONSPEECH label
2. Application of degradation (one axis at a time; others neutral)
3. Embedding in 2000 ms container with Gaussian noise padding (σ=10⁻⁴)
4. Query to model with prompt + normalization to SPEECH/NONSPEECH

**Example prompt shown:** LoRA+OPRO prompt (multi-line with definitions)

### Panel (b): Degradation Examples (4×2 Grid)
Four rows (one per degradation axis), two columns (neutral vs extreme):

| Axis | Neutral Condition | Extreme Condition |
|------|------------------|-------------------|
| **Duration** | 1000 ms | 20 ms |
| **SNR** | +20 dB (clean) | −10 dB |
| **Reverb** | RT60 = 0.0 s (none) | RT60 = 2.5 s (hall) |
| **Filter** | None | Bandpass 300–3400 Hz |

Each cell contains:
- Mini waveform plot
- Mini log-mel spectrogram
- Degradation value label
- Predictions: GT, Baseline, LoRA+OPRO

**Example selection strategy:**
- **Neutral:** Both Baseline and LoRA+OPRO correct (shows baseline competence)
- **Extreme:** Baseline fails, LoRA+OPRO succeeds (shows robustness improvement)

### Panel (c): 2×2 Experimental Design
Factorial design showing four configurations:

|              | Hand-crafted Prompt | OPRO Prompt |
|--------------|---------------------|-------------|
| **Base Weights** | Baseline | Base + OPRO |
| **LoRA Weights** | LoRA + Hand | LoRA + OPRO (Best) |

Note: Same evaluation pipeline; only prompt and/or weights change.

## Generated Artifacts

### Main Figure
- **PDF (publication-quality):** `figures/Fig_01_DegradationBank_Examples.pdf` (427 KB)
- **PNG (preview):** `figures/Fig_01_DegradationBank_Examples.png` (960 KB, 4520×3981 px, 300 DPI)

### Audio Examples
All examples are 16 kHz mono WAV files (2000 ms duration):

```
audio/Fig_01_duration_neutral.wav   # 1000 ms clip in 2000 ms container
audio/Fig_01_duration_extreme.wav   # 20 ms clip in 2000 ms container
audio/Fig_01_snr_neutral.wav        # +20 dB SNR
audio/Fig_01_snr_extreme.wav        # −10 dB SNR
audio/Fig_01_reverb_neutral.wav     # No reverb
audio/Fig_01_reverb_extreme.wav     # RT60 = 2.5 s
audio/Fig_01_filter_neutral.wav     # No filtering
audio/Fig_01_filter_extreme.wav     # Bandpass 300–3400 Hz
```

### Metadata Manifest
- **JSON:** `data/Fig_01_Examples_manifest.json`

Contains complete metadata for all examples:
- Audio paths (absolute and relative)
- Variant IDs and clip IDs
- Ground truth labels
- Predictions from all 4 configurations (Baseline, Base+OPRO, LoRA, LoRA+OPRO)
- Degradation conditions and axes

## Selected Examples (Seed 42)

### Duration
- **Neutral (1000 ms):** `esc50_1-58923-A-27_0368_1000ms_dur1000ms` - NONSPEECH (all correct)
- **Extreme (20 ms):** `voxconverse_kckqn_0291_1000ms_dur20ms` - SPEECH (Baseline→NONSPEECH ✗, LoRA+OPRO→SPEECH ✓)

### SNR
- **Neutral (+20 dB):** `esc50_3-141559-A-45_0482_1000ms_snr+20dB` - NONSPEECH (all correct)
- **Extreme (−10 dB):** `voxconverse_czlvt_0447_1000ms_snr-10dB` - SPEECH (Baseline→NONSPEECH ✗, LoRA+OPRO→SPEECH ✓)

### Reverb
- **Neutral (RT60=0.0):** `esc50_4-183487-A-1_0117_1000ms_reverbnone` - NONSPEECH (all correct)
- **Extreme (RT60=2.5):** `voxconverse_xvllq_0258_1000ms_reverb2.5s` - SPEECH (Baseline→NONSPEECH ✗, LoRA+OPRO→SPEECH ✓)

### Filter
- **Neutral (None):** `esc50_3-197408-C-8_0400_1000ms_filternone` - NONSPEECH (all correct)
- **Extreme (Bandpass):** `voxconverse_tiams_0443_1000ms_filterbandpass` - SPEECH (Baseline→NONSPEECH ✗, LoRA+OPRO→SPEECH ✓)

## Reproducibility

To regenerate this figure with different examples or settings:

```bash
# Edit parameters in the script if needed
python scripts/generate_figure1_degradation_bank.py

# Use a different random seed for example selection
# (edit the script: find_illustrative_examples(..., seed=YOUR_SEED))
```

All examples are deterministic given the same random seed (42).

## LaTeX Integration

To include this figure in your paper:

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{paper_artifacts/figures/Fig_01_DegradationBank_Examples.pdf}
  \caption{Psychometric degradation bank and experimental design.
    (a) Pipeline schematic showing degradation application and query workflow.
    (b) Representative examples for each degradation axis (neutral vs extreme).
    Each cell shows waveform, log-mel spectrogram, and model predictions (GT=ground truth, Base=Baseline, L+O=LoRA+OPRO).
    (c) 2×2 factorial design of experimental conditions.}
  \label{fig:degradation_bank}
\end{figure*}
```

## Figure Quality

- **Resolution:** 300 DPI (publication-ready)
- **Format:** PDF with vector graphics (text and diagrams) + embedded raster (spectrograms)
- **Size:** ~427 KB (PDF), well within typical journal limits
- **Dimensions:** 16" × 14" (suitable for full-width two-column layout)

## Notes

1. **Panel labels (a), (b), (c)** are embedded in the figure (no need for LaTeX subfigures)
2. **Color scheme:**
   - Panel (a) boxes: Light blue (clip), yellow (degradation), gray (container), green (query)
   - Panel (c) cells: Blue (Baseline), yellow (Base+OPRO), purple (LoRA), green (LoRA+OPRO - best)
3. **Font sizes:** Optimized for readability at full page width
4. **Spectrograms:** All use same dynamic range and color scale (viridis) for visual consistency

## Dependencies

- Python 3.11+
- matplotlib, numpy, pandas
- librosa (audio loading + spectrogram computation)
- soundfile (WAV export)

All dependencies are available in the `opro` conda environment.
