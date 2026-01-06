#!/usr/bin/env python3
"""
Generate Psychometric Threshold Artifacts (Results.5)

Creates:
- Tab_R04_PsychometricThresholds.csv/.tex
- Fig_R07a_DT90_ByConfig.pdf/.png
- Fig_R07b_SNR75_ByConfig.pdf/.png
- Fig_R07_Thresholds_DT90_SNR75.pdf (combined)
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Configuration order (as specified)
CONFIG_ORDER = [
    'baseline',
    'base_opro_classic',
    'lora_hand',
    'lora_opro_classic',
    'lora_opro_open'
]

# Display names
CONFIG_NAMES = {
    'baseline': 'Baseline',
    'base_opro_classic': 'Base+OPRO-C',
    'lora_hand': 'LoRA+Hand',
    'lora_opro_classic': 'LoRA+OPRO-C',
    'lora_opro_open': 'LoRA+OPRO-O'
}


def load_psychometric_data(json_path):
    """Load psychometric thresholds from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['configurations']


def create_threshold_table(configs):
    """Create table with DT50/75/90 and SNR75 thresholds."""

    rows = []

    for config_key in CONFIG_ORDER:
        if config_key not in configs:
            print(f"⚠️  Warning: {config_key} not found in results")
            continue

        config = configs[config_key]
        config_name = CONFIG_NAMES[config_key]

        row = {'Config': config_name}

        # Duration thresholds
        for dt_name in ['DT50', 'DT75', 'DT90']:
            dt_data = config['duration_thresholds'].get(dt_name, {})

            if dt_data:
                point = dt_data['point']
                ci_low, ci_high = dt_data['ci']
                censoring = dt_data.get('censoring', 'ok')

                row[f'{dt_name}_ms'] = point
                row[f'{dt_name}_CI_low'] = ci_low
                row[f'{dt_name}_CI_high'] = ci_high
                row[f'{dt_name}_flag'] = censoring
            else:
                row[f'{dt_name}_ms'] = None
                row[f'{dt_name}_CI_low'] = None
                row[f'{dt_name}_CI_high'] = None
                row[f'{dt_name}_flag'] = 'missing'

        # SNR threshold
        snr_data = config['snr_thresholds'].get('SNR75', {})
        if snr_data:
            point = snr_data['point']
            ci_low, ci_high = snr_data['ci']
            censoring = snr_data.get('censoring', 'ok')

            row['SNR75_dB'] = point
            row['SNR75_CI_low'] = ci_low
            row['SNR75_CI_high'] = ci_high
            row['SNR75_flag'] = censoring
        else:
            row['SNR75_dB'] = None
            row['SNR75_CI_low'] = None
            row['SNR75_CI_high'] = None
            row['SNR75_flag'] = 'missing'

        rows.append(row)

    return pd.DataFrame(rows)


def export_table_csv(df, output_path):
    """Export table as CSV."""
    df.to_csv(output_path, index=False)
    print(f"✓ Saved CSV: {output_path}")


def export_table_latex(df, output_path):
    """Export table as LaTeX (booktabs style)."""

    # Prepare formatted table for LaTeX
    latex_rows = []

    for _, row in df.iterrows():
        config = row['Config']

        # Format each threshold with CI
        dt50_str = f"{row['DT50_ms']:.1f} [{row['DT50_CI_low']:.1f}, {row['DT50_CI_high']:.1f}]"
        dt50_flag = f" ({row['DT50_flag']})" if row['DT50_flag'] != 'ok' else ""

        dt75_str = f"{row['DT75_ms']:.1f} [{row['DT75_CI_low']:.1f}, {row['DT75_CI_high']:.1f}]"
        dt75_flag = f" ({row['DT75_flag']})" if row['DT75_flag'] != 'ok' else ""

        dt90_str = f"{row['DT90_ms']:.1f} [{row['DT90_CI_low']:.1f}, {row['DT90_CI_high']:.1f}]"
        dt90_flag = f" ({row['DT90_flag']})" if row['DT90_flag'] != 'ok' else ""

        snr75_str = f"{row['SNR75_dB']:.1f} [{row['SNR75_CI_low']:.1f}, {row['SNR75_CI_high']:.1f}]"
        snr75_flag = f" ({row['SNR75_flag']})" if row['SNR75_flag'] != 'ok' else ""

        latex_rows.append(
            f"{config} & {dt50_str}{dt50_flag} & {dt75_str}{dt75_flag} & "
            f"{dt90_str}{dt90_flag} & {snr75_str}{snr75_flag} \\\\"
        )

    # Build LaTeX table
    latex = r"""\begin{table}[ht]
\centering
\caption{Psychometric Thresholds (DT50/75/90 and SNR-75) with Bootstrap CIs}
\label{tab:psychometric_thresholds}
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration} & \textbf{DT50 (ms)} & \textbf{DT75 (ms)} & \textbf{DT90 (ms)} & \textbf{SNR75 (dB)} \\
\midrule
"""
    latex += "\n".join(latex_rows)
    latex += r"""
\bottomrule
\end{tabular}
\vspace{0.5em}
\begin{flushleft}
\footnotesize
\textit{Note:} Values shown as point estimate [95\% CI]. Flags: (below\_range) = censored below, (above\_range) = censored above, (ok) = within range.
\end{flushleft}
\end{table}
"""

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"✓ Saved LaTeX: {output_path}")


def plot_dt90_comparison(df, output_base):
    """Create DT90 comparison figure."""

    fig, ax = plt.subplots(figsize=(8, 5))

    configs = df['Config'].values
    dt90_vals = df['DT90_ms'].values
    dt90_ci_low = df['DT90_CI_low'].values
    dt90_ci_high = df['DT90_CI_high'].values
    flags = df['DT90_flag'].values

    # Calculate error bars
    errors_low = dt90_vals - dt90_ci_low
    errors_high = dt90_ci_high - dt90_vals

    x = np.arange(len(configs))

    # Color by flag
    colors = []
    for flag in flags:
        if flag == 'ok':
            colors.append('#2ca02c')  # green
        elif flag == 'below_range':
            colors.append('#ff7f0e')  # orange
        elif flag == 'above_range':
            colors.append('#d62728')  # red
        else:
            colors.append('#7f7f7f')  # gray

    # Plot bars with error bars
    bars = ax.bar(x, dt90_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax.errorbar(x, dt90_vals, yerr=[errors_low, errors_high],
                fmt='none', ecolor='black', capsize=5, capthick=1.5)

    # Formatting
    ax.set_ylabel('DT90 (ms)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_title('Duration Threshold DT90 by Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Legend for flags
    legend_patches = [
        mpatches.Patch(color='#2ca02c', label='Within range'),
        mpatches.Patch(color='#ff7f0e', label='Below range'),
        mpatches.Patch(color='#d62728', label='Above range')
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=9)

    plt.tight_layout()

    # Save PDF and PNG
    pdf_path = f"{output_base}.pdf"
    png_path = f"{output_base}.png"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")


def plot_snr75_comparison(df, output_base):
    """Create SNR75 comparison figure."""

    fig, ax = plt.subplots(figsize=(8, 5))

    configs = df['Config'].values
    snr75_vals = df['SNR75_dB'].values
    snr75_ci_low = df['SNR75_CI_low'].values
    snr75_ci_high = df['SNR75_CI_high'].values
    flags = df['SNR75_flag'].values

    # Calculate error bars
    errors_low = snr75_vals - snr75_ci_low
    errors_high = snr75_ci_high - snr75_vals

    x = np.arange(len(configs))

    # Color by flag
    colors = []
    for flag in flags:
        if flag == 'ok':
            colors.append('#2ca02c')  # green
        elif flag == 'below_range':
            colors.append('#ff7f0e')  # orange
        elif flag == 'above_range':
            colors.append('#d62728')  # red
        else:
            colors.append('#7f7f7f')  # gray

    # Plot bars with error bars
    bars = ax.bar(x, snr75_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax.errorbar(x, snr75_vals, yerr=[errors_low, errors_high],
                fmt='none', ecolor='black', capsize=5, capthick=1.5)

    # Formatting
    ax.set_ylabel('SNR75 (dB)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_title('SNR Threshold (75% Accuracy) by Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Legend for flags
    legend_patches = [
        mpatches.Patch(color='#2ca02c', label='Within range'),
        mpatches.Patch(color='#ff7f0e', label='Below range'),
        mpatches.Patch(color='#d62728', label='Above range')
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=9)

    plt.tight_layout()

    # Save PDF and PNG
    pdf_path = f"{output_base}.pdf"
    png_path = f"{output_base}.png"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")


def create_combined_figure(df, output_path):
    """Create combined DT90 + SNR75 figure (vertical stack)."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    configs = df['Config'].values
    x = np.arange(len(configs))

    # Panel A: DT90
    dt90_vals = df['DT90_ms'].values
    dt90_ci_low = df['DT90_CI_low'].values
    dt90_ci_high = df['DT90_CI_high'].values
    dt90_flags = df['DT90_flag'].values

    errors_low_dt90 = dt90_vals - dt90_ci_low
    errors_high_dt90 = dt90_ci_high - dt90_vals

    colors_dt90 = []
    for flag in dt90_flags:
        if flag == 'ok':
            colors_dt90.append('#2ca02c')
        elif flag == 'below_range':
            colors_dt90.append('#ff7f0e')
        elif flag == 'above_range':
            colors_dt90.append('#d62728')
        else:
            colors_dt90.append('#7f7f7f')

    ax1.bar(x, dt90_vals, color=colors_dt90, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.errorbar(x, dt90_vals, yerr=[errors_low_dt90, errors_high_dt90],
                 fmt='none', ecolor='black', capsize=5, capthick=1.5)
    ax1.set_ylabel('DT90 (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('A) Duration Threshold DT90', fontsize=13, fontweight='bold', loc='left')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Panel B: SNR75
    snr75_vals = df['SNR75_dB'].values
    snr75_ci_low = df['SNR75_CI_low'].values
    snr75_ci_high = df['SNR75_CI_high'].values
    snr75_flags = df['SNR75_flag'].values

    errors_low_snr = snr75_vals - snr75_ci_low
    errors_high_snr = snr75_ci_high - snr75_vals

    colors_snr = []
    for flag in snr75_flags:
        if flag == 'ok':
            colors_snr.append('#2ca02c')
        elif flag == 'below_range':
            colors_snr.append('#ff7f0e')
        elif flag == 'above_range':
            colors_snr.append('#d62728')
        else:
            colors_snr.append('#7f7f7f')

    ax2.bar(x, snr75_vals, color=colors_snr, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.errorbar(x, snr75_vals, yerr=[errors_low_snr, errors_high_snr],
                 fmt='none', ecolor='black', capsize=5, capthick=1.5)
    ax2.set_ylabel('SNR75 (dB)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax2.set_title('B) SNR Threshold (75% Accuracy)', fontsize=13, fontweight='bold', loc='left')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    # Common legend
    legend_patches = [
        mpatches.Patch(color='#2ca02c', label='Within range'),
        mpatches.Patch(color='#ff7f0e', label='Below range'),
        mpatches.Patch(color='#d62728', label='Above range')
    ]
    fig.legend(handles=legend_patches, loc='upper right',
               bbox_to_anchor=(0.98, 0.98), fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save PDF
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")


def main():
    # Paths
    json_path = Path('paper_artifacts/data/psychometric_run/all_psychometric_thresholds.json')
    tables_dir = Path('paper_artifacts/tables')
    figures_dir = Path('paper_artifacts/figures')

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Generating Psychometric Threshold Artifacts (Results.5)")
    print("="*80)

    # Load data
    print(f"\n📊 Loading data from: {json_path}")
    configs = load_psychometric_data(json_path)

    # Create table
    print("\n📋 Creating threshold table...")
    df = create_threshold_table(configs)

    # Export table
    csv_path = tables_dir / 'Tab_R04_PsychometricThresholds.csv'
    tex_path = tables_dir / 'Tab_R04_PsychometricThresholds.tex'

    export_table_csv(df, csv_path)
    export_table_latex(df, tex_path)

    # Create figures
    print("\n📈 Creating figures...")

    # Individual figures
    plot_dt90_comparison(df, figures_dir / 'Fig_R07a_DT90_ByConfig')
    plot_snr75_comparison(df, figures_dir / 'Fig_R07b_SNR75_ByConfig')

    # Combined figure
    combined_path = figures_dir / 'Fig_R07_Thresholds_DT90_SNR75.pdf'
    create_combined_figure(df, combined_path)

    print("\n" + "="*80)
    print("✅ DONE - All artifacts generated")
    print("="*80)
    print("\nGenerated files:")
    print(f"  • {csv_path}")
    print(f"  • {tex_path}")
    print(f"  • {figures_dir}/Fig_R07a_DT90_ByConfig.pdf/.png")
    print(f"  • {figures_dir}/Fig_R07b_SNR75_ByConfig.pdf/.png")
    print(f"  • {combined_path}")


if __name__ == '__main__':
    main()
