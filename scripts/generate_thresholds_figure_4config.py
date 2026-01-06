#!/usr/bin/env python3
"""
Generate Fig_R07: Psychometric Thresholds (DT90 and SNR75) for 4-config design.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Publication settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Load statistical analysis results
STATS_FILE = Path('results/statistical_analysis/statistical_analysis.json')
OUTPUT_DIR = Path('figures')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

with open(STATS_FILE, 'r') as f:
    stats = json.load(f)

# Configuration order (4-config design)
CONFIG_ORDER = ['baseline', 'base_opro', 'lora', 'lora_opro_classic']

# Display names
CONFIG_NAMES = {
    'baseline': 'Baseline',
    'base_opro': 'Base+OPRO',
    'lora': 'LoRA',
    'lora_opro_classic': 'LoRA+OPRO'
}

# Colors
CONFIG_COLORS = {
    'baseline': '#e74c3c',
    'base_opro': '#f39c12',
    'lora': '#3498db',
    'lora_opro_classic': '#2ecc71'
}

print("="*80)
print("GENERATING FIG_R07: PSYCHOMETRIC THRESHOLDS")
print("="*80)

# Extract threshold data
thresholds = stats.get('psychometric_thresholds', {})

# Prepare data for plotting
dt90_values = []
dt90_ci_low = []
dt90_ci_high = []
snr75_values = []
snr75_ci_low = []
snr75_ci_high = []
config_labels = []

for config_key in CONFIG_ORDER:
    if config_key in thresholds:
        config_data = thresholds[config_key]

        # DT90
        dt90 = config_data.get('duration_thresholds', {}).get('DT90', {})
        if dt90:
            dt90_values.append(dt90['point'])
            dt90_ci_low.append(dt90['ci'][0])
            dt90_ci_high.append(dt90['ci'][1])
        else:
            dt90_values.append(np.nan)
            dt90_ci_low.append(np.nan)
            dt90_ci_high.append(np.nan)

        # SNR75
        snr75 = config_data.get('snr_thresholds', {}).get('SNR75', {})
        if snr75:
            snr75_values.append(snr75['point'])
            snr75_ci_low.append(snr75['ci'][0])
            snr75_ci_high.append(snr75['ci'][1])
        else:
            snr75_values.append(np.nan)
            snr75_ci_low.append(np.nan)
            snr75_ci_high.append(np.nan)

        config_labels.append(CONFIG_NAMES[config_key])
    else:
        print(f"⚠️  Warning: {config_key} not found in psychometric_thresholds")

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# === Subplot 1: DT90 ===
x = np.arange(len(CONFIG_ORDER))
bars1 = ax1.bar(x, dt90_values,
                color=[CONFIG_COLORS[c] for c in CONFIG_ORDER],
                edgecolor='black', linewidth=1.2, alpha=0.8)

# Error bars for DT90
errors_dt90 = [
    [dt90_values[i] - dt90_ci_low[i] if not np.isnan(dt90_values[i]) else 0 for i in range(len(CONFIG_ORDER))],
    [dt90_ci_high[i] - dt90_values[i] if not np.isnan(dt90_values[i]) else 0 for i in range(len(CONFIG_ORDER))]
]

ax1.errorbar(x, dt90_values, yerr=errors_dt90, fmt='none',
            ecolor='black', capsize=5, capthick=1.5, linewidth=1.5)

ax1.set_ylabel('DT90 (ms)', fontweight='bold')
ax1.set_xlabel('Configuration', fontweight='bold')
ax1.set_title('Duration Threshold (90% Accuracy)', fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(config_labels, rotation=15, ha='right')
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars1, dt90_values)):
    if not np.isnan(val):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + errors_dt90[1][i] + 5,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Add note
ax1.text(0.02, 0.98, 'Lower is better\n(more robust)',
        transform=ax1.transAxes, ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
        fontsize=9)

# === Subplot 2: SNR75 ===
bars2 = ax2.bar(x, snr75_values,
                color=[CONFIG_COLORS[c] for c in CONFIG_ORDER],
                edgecolor='black', linewidth=1.2, alpha=0.8)

# Error bars for SNR75
errors_snr75 = [
    [snr75_values[i] - snr75_ci_low[i] if not np.isnan(snr75_values[i]) else 0 for i in range(len(CONFIG_ORDER))],
    [snr75_ci_high[i] - snr75_values[i] if not np.isnan(snr75_values[i]) else 0 for i in range(len(CONFIG_ORDER))]
]

ax2.errorbar(x, snr75_values, yerr=errors_snr75, fmt='none',
            ecolor='black', capsize=5, capthick=1.5, linewidth=1.5)

ax2.set_ylabel('SNR75 (dB)', fontweight='bold')
ax2.set_xlabel('Configuration', fontweight='bold')
ax2.set_title('SNR Threshold (75% Accuracy)', fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(config_labels, rotation=15, ha='right')
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars2, snr75_values)):
    if not np.isnan(val):
        height = bar.get_height()
        y_pos = height + errors_snr75[1][i] + 0.5 if height >= 0 else height - errors_snr75[0][i] - 0.5
        va = 'bottom' if height >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.1f}', ha='center', va=va, fontweight='bold', fontsize=9)

# Add note
ax2.text(0.02, 0.98, 'Lower is better\n(more robust)',
        transform=ax2.transAxes, ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
        fontsize=9)

plt.tight_layout()

# Save figure
plt.savefig(OUTPUT_DIR / 'Fig_R07_Thresholds_DT90_SNR75.pdf')
plt.savefig(OUTPUT_DIR / 'Fig_R07_Thresholds_DT90_SNR75.png')
print(f"\n✓ Saved: Fig_R07_Thresholds_DT90_SNR75.pdf")
print(f"✓ Saved: Fig_R07_Thresholds_DT90_SNR75.png")
plt.close()

print("\n" + "="*80)
print("THRESHOLD FIGURE GENERATED SUCCESSFULLY")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}")
