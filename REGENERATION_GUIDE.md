# Results Regeneration Guide

This guide documents the steps to regenerate all tables and figures from the single source of truth: `results/complete_pipeline_seed42`.

## ✅ Completed Tasks

1. **Data Integrity Verification**
   - Verified all 4 configurations in `complete_pipeline_seed42` have consistent hashes
   - Confirmed single-run consistency
   - Report: `results/data_integrity_report.json`

2. **Main Document Updates**
   - Removed all "Open" variant references from Results section
   - Updated to 4-configuration design: Baseline, Base+OPRO, LoRA, LoRA+OPRO
   - Updated statistical comparisons (4 → 3 primary comparisons)
   - Created Appendix documenting Classic vs Open comparison

3. **Prompt Summary Table**
   - Generated `tables/Tab_PromptSummary.tex` with 4 configurations
   - Includes character counts and definitions indicator
   - Successfully integrated into main.tex

4. **Essential Tables**
   - `Tab_R02_OverallPerformance.tex` ✅ (copied from paper_artifacts)
   - `Tab_R05_ErrorCounts.tex` ✅ (copied from paper_artifacts)
   - `Tab_PromptSummary.tex` ✅ (generated)

## ⚠️  Pending: Figure Regeneration

The following figures need regeneration from `complete_pipeline_seed42`:

```
figures/Fig_R01_Overall_BAclip.png
figures/Fig_R02_DeltaBA_PrimaryComparisons.png
figures/Fig_R03_Duration_BAbyCondition.png
figures/Fig_R04_SNR_BAbyCondition.png
figures/Fig_R05_Reverb_BAbyCondition.png
figures/Fig_R06_Filter_BAbyCondition.png
figures/Fig_R07_Thresholds_DT90_SNR75.pdf
figures/Fig_R08_Recall_Tradeoff.png
```

### How to Regenerate Figures

**Option 1: Use existing scripts (requires adaptation)**

The following scripts exist but need updating to remove Open variant:

- `scripts/generate_overall_performance.py`
- `scripts/generate_results3_primary_comparisons.py`
- `scripts/generate_robustness_by_condition.py`
- `scripts/generate_psychometric_artifacts.py`
- `scripts/generate_error_profile.py`
- `scripts/generate_figures.py`

**Required changes:**
1. Update `MODEL_NAMES` dictionaries to use 4 configs only
2. Remove `lora_opro_open` from all lists
3. Update color schemes and legends
4. Save outputs with correct filenames (Fig_R01, Fig_R02, etc.)

**Option 2: Manual consolidation**

Some figures already exist in `results/figures/` with different names:
- `figure1_ba_comparison.png` → `Fig_R01_Overall_BAclip.png`
- `figure2_comparisons.png` → `Fig_R02_DeltaBA_PrimaryComparisons.png`
- `figure3_recall_tradeoff.png` → `Fig_R08_Recall_Tradeoff.png`

Check these files first and rename if they match the required content (4 configs only).

**Option 3: Create master regeneration script**

Create `scripts/regenerate_all_figures.sh`:

```bash
#!/bin/bash
# Regenerate all figures from complete_pipeline_seed42

set -e

RUN_DIR="results/complete_pipeline_seed42"
OUTPUT_DIR="figures"

mkdir -p "$OUTPUT_DIR"

# Run each figure generation script
python scripts/generate_overall_performance.py --run-dir "$RUN_DIR" --output "$OUTPUT_DIR"
python scripts/generate_results3_primary_comparisons.py --run-dir "$RUN_DIR" --output "$OUTPUT_DIR"
python scripts/generate_robustness_by_condition.py --run-dir "$RUN_DIR" --output "$OUTPUT_DIR"
python scripts/generate_psychometric_artifacts.py --run-dir "$RUN_DIR" --output "$OUTPUT_DIR"
python scripts/generate_error_profile.py --run-dir "$RUN_DIR" --output "$OUTPUT_DIR"

echo "All figures regenerated successfully"
```

## Verification Checklist

After regenerating figures, run:

```bash
python scripts/generate_final_verification_log.py
```

Expected output:
- ✅ All 4 configs have consistent hashes
- ✅ No Open references in Results section
- ✅ All tables present
- ✅ All figures present

## Data Integrity Confirmation

**Source of Truth:** `results/complete_pipeline_seed42`

**Configurations:**
- `01_baseline` (hash: ba6458496986a425...)
- `06_eval_base_opro` (hash: c771ef8eab0fd18d...)
- `03_eval_lora` (hash: bc7a5018a7be9fed...)
- `07_eval_lora_opro` (hash: 4966826cb554926d...)

**Test Set:**
- 970 base clips × 22 conditions = 21,340 samples
- Perfect class balance: 485 SPEECH + 485 NONSPEECH per condition

## Appendix: Classic vs Open

The Appendix (Section A.1) documents:
- Experimental setup differences
- Discordant pair counts (27.45% for Base+OPRO, 5.78% for LoRA+OPRO)
- Rationale for excluding Open from main results

This preserves the experimental record while maintaining clean 4-config narrative in Results.

## Next Steps

1. Adapt figure generation scripts to 4-config design
2. Regenerate all 8 figures
3. Run final verification
4. Compile paper and check all cross-references

---

**Last Updated:** 2025-12-30
**Status:** Main document clean, figures pending regeneration
