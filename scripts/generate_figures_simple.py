#!/usr/bin/env python3
"""
Generate publication-quality figures - SIMPLIFIED VERSION
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

# Paths
RESULTS_DIR = Path('/mnt/fast/nobackup/users/gb0048/opro2_clean/results')
OUTPUT_DIR = RESULTS_DIR / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Load data
with open(RESULTS_DIR / 'statistical_analysis/statistical_analysis.json', 'r') as f:
    data = json.load(f)

print("="*80)
print("GENERATING FIGURES")
print("="*80)

# Figure 1: BA Comparison
print("\n[1/3] Generating BA comparison...")
configs = data['config_metrics']
models = ['baseline', 'base_opro', 'lora', 'lora_opro_classic']
names = ['Baseline', 'Base+OPRO', 'LoRA+Hand', 'LoRA+OPRO']
colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

ba_vals = [configs[m]['ba_clip'] for m in models]
ba_ci_low = [configs[m]['ba_clip_ci'][0] for m in models]
ba_ci_high = [configs[m]['ba_clip_ci'][1] for m in models]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(models))
bars = ax.bar(x, ba_vals, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
ax.errorbar(x, ba_vals,
            yerr=[[ba_vals[i] - ba_ci_low[i] for i in range(len(models))],
                  [ba_ci_high[i] - ba_vals[i] for i in range(len(models))]],
            fmt='none', ecolor='black', capsize=5, capthick=1.5, linewidth=1.5)

ax.set_ylabel('Balanced Accuracy (BA_clip)', fontweight='bold')
ax.set_xlabel('Model', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha='right')
ax.set_ylim([0.5, 1.0])
ax.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, ba_vals)):
    ax.text(bar.get_x() + bar.get_width()/2., val + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

ax.set_title('Balanced Accuracy by Model', fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure1_ba_comparison.png')
plt.savefig(OUTPUT_DIR / 'figure1_ba_comparison.pdf')
print(f"  ✓ Saved: figure1_ba_comparison.png")
plt.close()

# Figure 2: Comparisons
print("\n[2/3] Generating comparisons forest plot...")
comps = data['primary_comparisons']
comp_names = list(comps.keys())
deltas = [comps[c]['delta_ba'] for c in comp_names]
ci_low = [comps[c]['delta_ba_ci'][0] for c in comp_names]
ci_high = [comps[c]['delta_ba_ci'][1] for c in comp_names]
pvals = [comps[c]['p_value_adjusted'] for c in comp_names]

fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(comp_names))
colors_sig = ['green' if p < 0.05 else 'gray' for p in pvals]

# Plot error bars and points individually for each comparison
for i in range(len(comp_names)):
    xerr = [[deltas[i] - ci_low[i]], [ci_high[i] - deltas[i]]]
    ax.errorbar(deltas[i], y[i], xerr=xerr,
                fmt='none', capsize=6, capthick=2,
                ecolor=colors_sig[i], linewidth=2)
    ax.plot(deltas[i], y[i], 'o', markersize=12, color=colors_sig[i],
            markeredgecolor='black', markeredgewidth=1.5)

ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax.set_yticks(y)
ax.set_yticklabels([c.replace(' vs ', '\nvs\n') for c in comp_names])
ax.set_xlabel('Difference in BA (ΔBA)', fontweight='bold')
ax.set_title('Pairwise Comparisons (95% CI)', fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)

for i, p in enumerate(pvals):
    p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(ax.get_xlim()[1] * 0.9, i, p_text, ha='right', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure2_comparisons.png')
plt.savefig(OUTPUT_DIR / 'figure2_comparisons.pdf')
print(f"  ✓ Saved: figure2_comparisons.png")
plt.close()

# Figure 3: Recall trade-off
print("\n[3/3] Generating recall trade-off plot...")
fig, ax = plt.subplots(figsize=(7, 7))

recall_speech = [configs[m]['recall_speech'] for m in models]
recall_nonspeech = [configs[m]['recall_nonspeech'] for m in models]

for i, model in enumerate(models):
    ax.scatter(recall_nonspeech[i], recall_speech[i],
               s=200, color=colors[i], edgecolor='black', linewidth=2,
               label=names[i], zorder=3)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
ax.set_xlabel('Recall (NonSpeech)', fontweight='bold')
ax.set_ylabel('Recall (Speech)', fontweight='bold')
ax.set_xlim([0.2, 1.0])
ax.set_ylim([0.2, 1.0])
ax.set_aspect('equal')
ax.legend(loc='lower right', framealpha=0.95)
ax.grid(alpha=0.3)
ax.set_title('Recall Trade-off', fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure3_recall_tradeoff.png')
plt.savefig(OUTPUT_DIR / 'figure3_recall_tradeoff.pdf')
print(f"  ✓ Saved: figure3_recall_tradeoff.png")
plt.close()

print("\n" + "="*80)
print("ALL FIGURES GENERATED")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nFiles generated:")
for f in sorted(OUTPUT_DIR.glob('*.png')):
    print(f"  - {f.name}")
