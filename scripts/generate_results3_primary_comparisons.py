#!/usr/bin/env python3
"""
Generate Results.3 artifacts: Primary Paired Comparisons table and figure
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load statistical analysis results
with open('results/statistical_analysis/statistical_analysis.json', 'r') as f:
    stats = json.load(f)

# Define the 4 primary comparisons in order
# Maps: json_key -> display_name
comparisons_map = {
    'Baseline vs Base+OPRO': 'Baseline vs Base+OPRO (Classic)',
    'Baseline vs LoRA+BasePrompt': 'Baseline vs LoRA (Hand-crafted)',
    'LoRA+BasePrompt vs LoRA+OPRO': 'LoRA vs LoRA+OPRO (Classic)',
    'LoRA+OPRO_Classic vs LoRA+OPRO_Open': 'LoRA+OPRO Classic vs LoRA+OPRO Open'
}

# Extract data for each comparison
rows = []
for json_name, display_name in comparisons_map.items():
    comp = stats['primary_comparisons'][json_name]

    row = {
        'Comparison': display_name,
        'Delta_BA': comp['delta_ba'],
        'CI_Low': comp['delta_ba_ci'][0],
        'CI_High': comp['delta_ba_ci'][1],
        'p_raw': comp['p_value_raw'],
        'p_adjusted': comp['p_value_adjusted'],
        'Significant': comp['significant'],
        'Discordant_Rate': comp['mcnemar']['discordant_rate'],
        'n_01': comp['mcnemar']['n_01'],
        'n_10': comp['mcnemar']['n_10']
    }
    rows.append(row)

df = pd.DataFrame(rows)

# Save CSV
csv_path = 'paper_artifacts/tables/Tab_R03_PrimaryComparisons.csv'
df.to_csv(csv_path, index=False)
print(f"✓ Saved: {csv_path}")

# Generate LaTeX table
def format_pvalue(p):
    """Format p-value in scientific notation when < 0.001"""
    if p < 0.001:
        return f"{p:.2e}"
    else:
        return f"{p:.3f}"

latex_rows = []
for _, row in df.iterrows():
    comp_name = row['Comparison'].replace('&', r'\&')
    delta_ba = f"{row['Delta_BA']:.4f}"
    ci = f"[{row['CI_Low']:.4f}, {row['CI_High']:.4f}]"
    p_raw = format_pvalue(row['p_raw'])
    p_adj = format_pvalue(row['p_adjusted'])
    sig = 'Yes' if row['Significant'] else 'No'
    disc_rate = f"{row['Discordant_Rate']:.3f}"
    n01_n10 = f"{int(row['n_01'])}/{int(row['n_10'])}"

    latex_rows.append(f"    {comp_name} & {delta_ba} & {ci} & {p_raw} & {p_adj} & {sig} & {disc_rate} & {n01_n10} \\\\")

latex_content = r"""\begin{table}[htbp]
\centering
\caption{Primary Paired Comparisons: Statistical Significance Tests}
\label{tab:primary_comparisons}
\begin{tabular}{lcccccccc}
\toprule
Comparison & $\Delta$BA & 95\% CI & $p$ (raw) & $p$ (Holm) & Sig. & Disc. Rate & $n_{01}$/$n_{10}$ \\
\midrule
""" + "\n".join(latex_rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: $\Delta$BA = BA(A) - BA(B). McNemar's exact test with Holm correction.
\item Discordant Rate = proportion of discordant pairs. $n_{01}$ = A incorrect, B correct; $n_{10}$ = A correct, B incorrect.
\end{tablenotes}
\end{table}
"""

tex_path = 'paper_artifacts/tables/Tab_R03_PrimaryComparisons.tex'
with open(tex_path, 'w') as f:
    f.write(latex_content)
print(f"✓ Saved: {tex_path}")

# Generate figure: Delta BA with error bars
fig, ax = plt.subplots(figsize=(10, 6))

# Extract data for plotting
comparison_labels = df['Comparison'].tolist()
delta_ba = df['Delta_BA'].values
ci_low = df['CI_Low'].values
ci_high = df['CI_High'].values
significant = df['Significant'].values

# Calculate error bars (distance from point estimate to CI bounds)
err_low = delta_ba - ci_low
err_high = ci_high - delta_ba

# Create horizontal bar plot
y_pos = np.arange(len(comparison_labels))
colors = ['#d62728' if sig else '#1f77b4' for sig in significant]

ax.barh(y_pos, delta_ba, xerr=[err_low, err_high],
        color=colors, alpha=0.7, capsize=5, ecolor='black')

# Add vertical line at 0
ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(comparison_labels, fontsize=10)
ax.set_xlabel('Δ Balanced Accuracy (A - B)', fontsize=12)
ax.set_title('Primary Comparisons: Δ BA with 95% CI', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d62728', alpha=0.7, label='Significant (p < 0.05)'),
    Patch(facecolor='#1f77b4', alpha=0.7, label='Not Significant')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()

# Save as PDF and PNG
pdf_path = 'paper_artifacts/figures/Fig_R02_DeltaBA_PrimaryComparisons.pdf'
png_path = 'paper_artifacts/figures/Fig_R02_DeltaBA_PrimaryComparisons.png'
plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {pdf_path}")
print(f"✓ Saved: {png_path}")

plt.close()

print("\n=== Summary ===")
print(f"Generated artifacts for {len(df)} primary comparisons")
print(f"- {sum(df['Significant'])} significant")
print(f"- {len(df) - sum(df['Significant'])} not significant")
