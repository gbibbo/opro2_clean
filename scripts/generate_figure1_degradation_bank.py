#!/usr/bin/env python3
"""
Generate Figure 1: Degradation Bank and Experimental Design

Creates a multi-panel figure showing:
(a) Pipeline schematic with example query
(b) 4×2 grid of examples (neutral vs extreme) for each degradation axis
(c) 2×2 experimental design diagram

Outputs:
- paper_artifacts/figures/Fig_01_DegradationBank_Examples.{pdf,png}
- paper_artifacts/data/Fig_01_Examples_manifest.json
- paper_artifacts/audio/Fig_01_<axis>_{neutral,extreme}.wav
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import librosa
import librosa.display
import soundfile as sf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path("/mnt/fast/nobackup/users/gb0048/opro2_clean")
DATA_ROOT = PROJECT_ROOT / "data/processed/variants_validated_1000"
RESULTS_ROOT = PROJECT_ROOT / "results/complete_pipeline_seed42"
OUTPUT_ROOT = PROJECT_ROOT / "paper_artifacts"

# Ensure output directories exist
(OUTPUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "data").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "audio").mkdir(parents=True, exist_ok=True)

# Prompts for each configuration
PROMPTS = {
    "baseline": "Does this audio contain human speech? Answer SPEECH or NONSPEECH.",
    "base_opro": "Listen briefly; is this clip human speech or noise? Quickly reply: SPEECH or NONSPEECH.",
    "lora": "Does this audio contain human speech? Answer SPEECH or NONSPEECH.",
    "lora_opro": """Decide the dominant content.
Definitions:
- SPEECH = human voice, spoken words, syllables, conversational cues.
- NONSPEECH = music, tones/beeps, environmental noise, silence.
Output exactly: SPEECH or NONSPEECH."""
}

# Degradation parameters
DEGRADATION_SPECS = {
    "duration": {
        "neutral": {"value": 1000, "condition": "dur_1000ms", "label": "1000 ms"},
        "extreme": {"value": 20, "condition": "dur_20ms", "label": "20 ms"}
    },
    "snr": {
        "neutral": {"value": 20, "condition": "snr_20dB", "label": "+20 dB"},
        "extreme": {"value": -10, "condition": "snr_-10dB", "label": "−10 dB"}
    },
    "reverb": {
        "neutral": {"value": 0.0, "condition": "reverb_none", "label": "RT60 = 0.0 s"},
        "extreme": {"value": 2.5, "condition": "reverb_2.5s", "label": "RT60 = 2.5 s"}
    },
    "filter": {
        "neutral": {"value": "none", "condition": "filter_none", "label": "None"},
        "extreme": {"value": "bandpass", "condition": "filter_bandpass", "label": "Bandpass 300–3400 Hz"}
    }
}


# =============================================================================
# Data Loading
# =============================================================================

def load_predictions():
    """Load predictions from all 4 configurations."""
    configs = {
        "baseline": RESULTS_ROOT / "01_baseline/predictions.csv",
        "base_opro": RESULTS_ROOT / "06_eval_base_opro/predictions.csv",
        "lora": RESULTS_ROOT / "03_eval_lora/predictions.csv",
        "lora_opro": RESULTS_ROOT / "07_eval_lora_opro/predictions.csv"
    }

    dfs = {}
    for name, path in configs.items():
        df = pd.read_csv(path)
        # Add absolute paths
        df['audio_path'] = df['audio_path'].apply(
            lambda x: str(PROJECT_ROOT / x) if not x.startswith('/') else x
        )
        dfs[name] = df

    return dfs


def load_metadata():
    """Load metadata for all test samples."""
    metadata_path = DATA_ROOT / "test_metadata.csv"
    df = pd.read_csv(metadata_path)

    # Add absolute paths
    df['audio_path'] = df['audio_path'].apply(
        lambda x: str(PROJECT_ROOT / x) if not x.startswith('/') else x
    )

    return df


# =============================================================================
# Example Selection
# =============================================================================

def find_illustrative_examples(predictions_dfs, metadata_df, seed=42):
    """
    Find illustrative examples where Baseline fails and LoRA+OPRO succeeds.

    Returns a dict with one example per axis: {axis: {neutral: {...}, extreme: {...}}}
    """
    np.random.seed(seed)

    baseline = predictions_dfs["baseline"]
    lora_opro = predictions_dfs["lora_opro"]

    # Merge predictions
    merged = baseline.merge(
        lora_opro,
        on="audio_path",
        suffixes=("_baseline", "_lora_opro")
    )

    examples = {}

    for axis, specs in DEGRADATION_SPECS.items():
        examples[axis] = {}

        for level in ["neutral", "extreme"]:
            condition = specs[level]["condition"]

            # Filter by condition
            candidates = merged[merged[f"condition_baseline"] == condition].copy()

            # For extreme: prefer Baseline wrong + LoRA+OPRO correct
            if level == "extreme":
                # First try: Baseline wrong + LoRA+OPRO correct
                error_to_correct = candidates[
                    (candidates["prediction_baseline"] != candidates["ground_truth_baseline"]) &
                    (candidates["prediction_lora_opro"] == candidates["ground_truth_lora_opro"])
                ]
                if len(error_to_correct) > 0:
                    candidates = error_to_correct
                else:
                    # Fallback: just LoRA+OPRO correct
                    candidates = candidates[
                        candidates["prediction_lora_opro"] == candidates["ground_truth_lora_opro"]
                    ]
            else:
                # For neutral: both models correct (show baseline performance)
                both_correct = candidates[
                    (candidates["prediction_baseline"] == candidates["ground_truth_baseline"]) &
                    (candidates["prediction_lora_opro"] == candidates["ground_truth_lora_opro"])
                ]
                if len(both_correct) > 0:
                    candidates = both_correct

            # Pick one randomly
            if len(candidates) > 0:
                idx = np.random.choice(len(candidates))
                sample = candidates.iloc[idx]

                # Get metadata
                meta = metadata_df[metadata_df["audio_path"] == sample["audio_path"]].iloc[0]

                examples[axis][level] = {
                    "audio_path": sample["audio_path"],
                    "variant_id": meta["variant_id"],
                    "clip_id": meta["clip_id"],
                    "ground_truth": sample["ground_truth_baseline"],
                    "condition": condition,
                    "axis": axis,
                    "level": level,
                    "predictions": {
                        "baseline": sample["prediction_baseline"],
                        "base_opro": None,  # Will fill if needed
                        "lora": None,
                        "lora_opro": sample["prediction_lora_opro"]
                    }
                }
            else:
                print(f"WARNING: No examples found for {axis}/{level}")

    # Fill in other predictions if available
    for axis in examples:
        for level in examples[axis]:
            audio_path = examples[axis][level]["audio_path"]

            for config in ["base_opro", "lora"]:
                match = predictions_dfs[config][predictions_dfs[config]["audio_path"] == audio_path]
                if len(match) > 0:
                    examples[axis][level]["predictions"][config] = match.iloc[0]["prediction"]

    return examples


# =============================================================================
# Audio Visualization
# =============================================================================

def plot_waveform_and_spectrogram(ax_wave, ax_spec, audio_path, title="",
                                   sr=16000, n_fft=512, hop_length=128):
    """Plot waveform and log-mel spectrogram in two subplots."""
    # Load audio
    y, _ = librosa.load(audio_path, sr=sr)

    # Waveform
    time = np.arange(len(y)) / sr
    ax_wave.plot(time, y, color='#2E86AB', linewidth=0.5, alpha=0.8)
    ax_wave.set_xlim(0, len(y)/sr)
    ax_wave.set_ylim(-1, 1)
    ax_wave.set_ylabel('Amplitude', fontsize=7)
    ax_wave.set_xlabel('Time (s)', fontsize=7)
    ax_wave.tick_params(labelsize=6)
    ax_wave.grid(True, alpha=0.3, linewidth=0.5)
    ax_wave.set_title(title, fontsize=8, fontweight='bold')

    # Log-mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=64
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    img = librosa.display.specshow(
        log_mel, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel',
        ax=ax_spec, cmap='viridis'
    )
    ax_spec.set_xlabel('Time (s)', fontsize=7)
    ax_spec.set_ylabel('Mel frequency', fontsize=7)
    ax_spec.tick_params(labelsize=6)

    return ax_wave, ax_spec


# =============================================================================
# Panel (a): Pipeline Schematic
# =============================================================================

def create_panel_a(ax):
    """Create pipeline schematic diagram."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(0.5, 7.5, '(a) Pipeline Schematic', fontsize=11, fontweight='bold',
            ha='left', va='top')

    # Box 1: Base clip
    rect1 = mpatches.FancyBboxPatch(
        (0.5, 5.5), 2.5, 1.5, boxstyle="round,pad=0.1",
        edgecolor='black', facecolor='#E8F4F8', linewidth=1.5
    )
    ax.add_patch(rect1)
    ax.text(1.75, 6.6, 'Base Clip', fontsize=9, ha='center', fontweight='bold')
    ax.text(1.75, 6.1, '1000 ms', fontsize=7, ha='center')
    ax.text(1.75, 5.8, 'Label: SPEECH/', fontsize=6, ha='center')
    ax.text(1.75, 5.6, 'NONSPEECH', fontsize=6, ha='center')

    # Arrow
    ax.annotate('', xy=(3.5, 6.25), xytext=(3.1, 6.25),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Box 2: Degradation
    rect2 = mpatches.FancyBboxPatch(
        (3.5, 5.5), 2.5, 1.5, boxstyle="round,pad=0.1",
        edgecolor='black', facecolor='#FFF4E6', linewidth=1.5
    )
    ax.add_patch(rect2)
    ax.text(4.75, 6.6, 'Apply Degradation', fontsize=9, ha='center', fontweight='bold')
    ax.text(4.75, 6.2, 'Duration / SNR /', fontsize=6, ha='center')
    ax.text(4.75, 5.95, 'Reverb / Filter', fontsize=6, ha='center')
    ax.text(4.75, 5.7, '(one axis at a time;', fontsize=6, ha='center', style='italic')
    ax.text(4.75, 5.5, 'others neutral)', fontsize=6, ha='center', style='italic')

    # Arrow
    ax.annotate('', xy=(6.5, 6.25), xytext=(6.1, 6.25),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Box 3: Container
    rect3 = mpatches.FancyBboxPatch(
        (6.5, 5.5), 2.5, 1.5, boxstyle="round,pad=0.1",
        edgecolor='black', facecolor='#F0F0F0', linewidth=1.5
    )
    ax.add_patch(rect3)
    ax.text(7.75, 6.6, 'Embed in Container', fontsize=9, ha='center', fontweight='bold')
    ax.text(7.75, 6.1, '2000 ms fixed', fontsize=7, ha='center')
    ax.text(7.75, 5.8, 'Padding: Gaussian', fontsize=6, ha='center')
    ax.text(7.75, 5.6, 'noise σ=10⁻⁴', fontsize=6, ha='center')

    # Arrow down
    ax.annotate('', xy=(7.75, 5.3), xytext=(7.75, 5.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Box 4: Query model
    rect4 = mpatches.FancyBboxPatch(
        (6.5, 3.5), 2.5, 1.5, boxstyle="round,pad=0.1",
        edgecolor='black', facecolor='#E8F8E8', linewidth=1.5
    )
    ax.add_patch(rect4)
    ax.text(7.75, 4.6, 'Query Model', fontsize=9, ha='center', fontweight='bold')
    ax.text(7.75, 4.2, 'with Prompt', fontsize=7, ha='center')
    ax.text(7.75, 3.9, '↓', fontsize=8, ha='center')
    ax.text(7.75, 3.7, 'Normalize to', fontsize=6, ha='center')
    ax.text(7.75, 3.55, 'SPEECH/NONSPEECH', fontsize=6, ha='center')

    # Example prompt (in a box below)
    prompt_box = mpatches.FancyBboxPatch(
        (0.5, 0.5), 8.5, 2.5, boxstyle="round,pad=0.1",
        edgecolor='#666', facecolor='#FFFEF0', linewidth=1, linestyle='--'
    )
    ax.add_patch(prompt_box)
    ax.text(1, 2.7, 'Example Prompt (LoRA+OPRO):', fontsize=7, ha='left',
            fontweight='bold', style='italic')
    prompt_text = PROMPTS["lora_opro"].replace('\n', '\n')
    y_pos = 2.3
    for line in prompt_text.split('\n'):
        ax.text(1, y_pos, line, fontsize=6, ha='left', family='monospace')
        y_pos -= 0.25

    return ax


# =============================================================================
# Panel (b): Examples Grid
# =============================================================================

def create_panel_b(fig, position, examples):
    """Create 4×2 grid of examples (neutral vs extreme for each axis)."""
    # Use GridSpec for finer control
    gs = GridSpec(4, 4, figure=fig,
                  left=position.x0, right=position.x1,
                  bottom=position.y0, top=position.y1,
                  hspace=0.8, wspace=0.4)

    axes_order = ["duration", "snr", "reverb", "filter"]
    row_labels = {
        "duration": "Duration",
        "snr": "SNR",
        "reverb": "Reverb",
        "filter": "Filter"
    }

    for i, axis in enumerate(axes_order):
        for j, level in enumerate(["neutral", "extreme"]):
            if axis not in examples or level not in examples[axis]:
                continue

            ex = examples[axis][level]

            # Column index: neutral=0,1 (wave, spec), extreme=2,3
            col_offset = 0 if level == "neutral" else 2

            # Create subplots for waveform and spectrogram
            ax_wave = fig.add_subplot(gs[i, col_offset])
            ax_spec = fig.add_subplot(gs[i, col_offset + 1])

            # Title with degradation value
            title = f"{row_labels[axis]}: {DEGRADATION_SPECS[axis][level]['label']}"

            # Plot
            plot_waveform_and_spectrogram(
                ax_wave, ax_spec, ex["audio_path"], title=title
            )

            # Add predictions as text below
            pred_text = (
                f"GT: {ex['ground_truth'][:6]}  "
                f"Base: {ex['predictions']['baseline'][:6]}  "
                f"L+O: {ex['predictions']['lora_opro'][:6]}"
            )
            ax_spec.text(0.5, -0.35, pred_text, fontsize=5, ha='center',
                        transform=ax_spec.transAxes)

    # Add main title
    fig.text(position.x0 + 0.02, position.y1 - 0.01,
             '(b) Degradation Examples: Neutral vs Extreme',
             fontsize=11, fontweight='bold', va='top')

    return gs


# =============================================================================
# Panel (c): 2×2 Design Diagram
# =============================================================================

def create_panel_c(ax):
    """Create 2×2 experimental design diagram."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(0.5, 5.5, '(c) Experimental Design: 2×2 Factorial',
            fontsize=11, fontweight='bold', ha='left', va='top')

    # Grid labels
    ax.text(1.5, 4.5, 'Prompt →', fontsize=8, ha='center', fontweight='bold')
    ax.text(0.3, 3, 'Weights', fontsize=8, ha='center', fontweight='bold', rotation=90)
    ax.text(0.3, 2.8, '↓', fontsize=9, ha='center')

    ax.text(4, 4.8, 'Hand-crafted', fontsize=8, ha='center', fontweight='bold')
    ax.text(7, 4.8, 'OPRO', fontsize=8, ha='center', fontweight='bold')
    ax.text(1.8, 3.5, 'Base', fontsize=8, ha='center', fontweight='bold')
    ax.text(1.8, 1.5, 'LoRA', fontsize=8, ha='center', fontweight='bold')

    # Cells
    cells = [
        # (x, y, width, height, label, color)
        (2.5, 2.7, 2.5, 1.3, 'Baseline\n(Base + Hand)', '#E3F2FD'),
        (5.5, 2.7, 2.5, 1.3, 'Base + OPRO', '#FFF9C4'),
        (2.5, 0.7, 2.5, 1.3, 'LoRA + Hand', '#F3E5F5'),
        (5.5, 0.7, 2.5, 1.3, 'LoRA + OPRO\n(Best)', '#C8E6C9'),
    ]

    for x, y, w, h, label, color in cells:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05",
            edgecolor='black', facecolor=color, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, fontsize=8, ha='center', va='center',
                fontweight='bold')

    # Note at bottom
    ax.text(5, 0.2,
            'Note: Same evaluation pipeline; only prompt and/or weights change',
            fontsize=6, ha='center', style='italic', color='#555')

    return ax


# =============================================================================
# Main Figure Generation
# =============================================================================

def generate_figure1(examples, output_prefix):
    """Generate complete Figure 1 with panels (a), (b), (c)."""
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 14))

    # Define positions manually for better control
    # Panel (a): top-left, landscape
    pos_a = fig.add_axes([0.05, 0.70, 0.45, 0.25])

    # Panel (c): top-right
    pos_c = fig.add_axes([0.55, 0.70, 0.40, 0.25])

    # Panel (b): bottom, full width
    # We'll use GridSpec inside for the 4×4 grid
    from matplotlib.transforms import Bbox
    pos_b = Bbox.from_bounds(0.05, 0.05, 0.90, 0.60)

    # Create panels
    create_panel_a(pos_a)
    create_panel_b(fig, pos_b, examples)
    create_panel_c(pos_c)

    # Save
    plt.savefig(f"{output_prefix}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Figure saved: {output_prefix}.pdf/png")


# =============================================================================
# Export Artifacts
# =============================================================================

def export_audio_examples(examples, output_dir):
    """Copy selected audio examples to output directory."""
    for axis in examples:
        for level in examples[axis]:
            ex = examples[axis][level]
            src = ex["audio_path"]
            dst = output_dir / f"Fig_01_{axis}_{level}.wav"

            # Copy using soundfile to ensure format consistency
            y, sr = librosa.load(src, sr=None)
            sf.write(dst, y, sr)

            print(f"✓ Exported: {dst.name}")


def save_manifest(examples, output_path):
    """Save JSON manifest with all example metadata."""
    manifest = {
        "figure_id": "Fig_01",
        "title": "Degradation Bank and Experimental Design",
        "date_generated": pd.Timestamp.now().isoformat(),
        "examples": {}
    }

    for axis in examples:
        manifest["examples"][axis] = {}
        for level in examples[axis]:
            ex = examples[axis][level].copy()
            # Convert Path objects to strings
            ex["audio_path"] = str(ex["audio_path"])
            manifest["examples"][axis][level] = ex

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Manifest saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("Generating Figure 1: Degradation Bank and Experimental Design")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading predictions and metadata...")
    predictions_dfs = load_predictions()
    metadata_df = load_metadata()
    print(f"  ✓ Loaded {len(predictions_dfs)} configurations")
    print(f"  ✓ Loaded metadata for {len(metadata_df)} samples")

    # Find examples
    print("\n[2/5] Finding illustrative examples...")
    examples = find_illustrative_examples(predictions_dfs, metadata_df, seed=42)
    for axis in examples:
        for level in examples[axis]:
            ex = examples[axis][level]
            print(f"  {axis}/{level}: {ex['variant_id']}")
            print(f"    GT={ex['ground_truth']}, "
                  f"Baseline={ex['predictions']['baseline']}, "
                  f"LoRA+OPRO={ex['predictions']['lora_opro']}")

    # Export audio
    print("\n[3/5] Exporting audio examples...")
    export_audio_examples(examples, OUTPUT_ROOT / "audio")

    # Save manifest
    print("\n[4/5] Saving manifest...")
    save_manifest(examples, OUTPUT_ROOT / "data/Fig_01_Examples_manifest.json")

    # Generate figure
    print("\n[5/5] Generating multi-panel figure...")
    output_prefix = OUTPUT_ROOT / "figures/Fig_01_DegradationBank_Examples"
    generate_figure1(examples, output_prefix)

    print("\n" + "=" * 80)
    print("✓ Figure 1 generation complete!")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  - Figure: {output_prefix}.pdf/png")
    print(f"  - Manifest: {OUTPUT_ROOT / 'data/Fig_01_Examples_manifest.json'}")
    print(f"  - Audio: {OUTPUT_ROOT / 'audio/Fig_01_*.wav'}")
    print()


if __name__ == "__main__":
    main()
