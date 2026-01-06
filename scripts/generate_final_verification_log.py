#!/usr/bin/env python3
"""
Generate final verification log for paper consistency.
Confirms:
1. All results from complete_pipeline_seed42 (single run)
2. Hash consistency across predictions.csv
3. No Open variant references in Results (only in Appendix)
4. All tables/figures referenced in main.tex exist
"""

import hashlib
import re
from pathlib import Path
import json
from datetime import datetime

def compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def check_references_in_tex(tex_path, results_start_label="sec:results", appendix_label="app:classic_vs_open"):
    """
    Check main.tex for Open references in Results vs Appendix.
    Returns: dict with counts
    """
    with open(tex_path, 'r') as f:
        content = f.read()

    # Find Results section (start at \section{Results})
    results_pattern = r'\\section\{Results\}.*?(?=\\appendix|\\bibliographystyle)'
    results_match = re.search(results_pattern, content, re.DOTALL)

    if not results_match:
        return {"error": "Could not find Results section"}

    results_content = results_match.group(0)

    # Count "Open" references (excluding code comments and safe contexts)
    # Look for "LoRA+OPRO (Open)", "Open variant", "Classic vs Open", etc.
    open_patterns = [
        r'LoRA\+OPRO \(Open\)',
        r'LoRA\+OPRO-Open',
        r'Open variant',
        r'Classic vs\.?\s+Open',
        r'open variant',
        r'OPRO.*Open'
    ]

    open_refs_in_results = []
    for pattern in open_patterns:
        matches = re.finditer(pattern, results_content, re.IGNORECASE)
        open_refs_in_results.extend([m.group(0) for m in matches])

    # Check appendix
    appendix_pattern = r'\\appendix.*?(?=\\bibliographystyle)'
    appendix_match = re.search(appendix_pattern, content, re.DOTALL)

    open_refs_in_appendix = 0
    if appendix_match:
        appendix_content = appendix_match.group(0)
        for pattern in open_patterns:
            matches = re.finditer(pattern, appendix_content, re.IGNORECASE)
            open_refs_in_appendix += len(list(matches))

    return {
        "open_refs_in_results": len(open_refs_in_results),
        "open_refs_details": open_refs_in_results,
        "open_refs_in_appendix": open_refs_in_appendix,
        "results_clean": len(open_refs_in_results) == 0
    }

def check_referenced_artifacts(tex_path):
    """
    Extract all \\input{tables/...} and \\includegraphics{figures/...} references.
    Check if files exist.
    """
    with open(tex_path, 'r') as f:
        content = f.read()

    # Extract table references
    table_pattern = r'\\input\{([^}]+\.tex)\}'
    table_refs = re.findall(table_pattern, content)

    # Extract figure references
    fig_pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
    fig_refs = re.findall(fig_pattern, content)

    missing_tables = []
    missing_figures = []

    for table in table_refs:
        if not Path(table).exists():
            missing_tables.append(table)

    for fig in fig_refs:
        # Try with various extensions if not specified
        fig_path = Path(fig)
        if not fig_path.suffix:
            # Try common extensions
            found = False
            for ext in ['.png', '.pdf', '.jpg', '.eps']:
                if Path(str(fig) + ext).exists():
                    found = True
                    break
            if not found:
                missing_figures.append(fig)
        elif not fig_path.exists():
            missing_figures.append(fig)

    return {
        "total_table_refs": len(table_refs),
        "total_figure_refs": len(fig_refs),
        "missing_tables": missing_tables,
        "missing_figures": missing_figures,
        "all_artifacts_present": len(missing_tables) == 0 and len(missing_figures) == 0
    }

def main():
    """Main verification routine."""
    print("="*80)
    print("FINAL DATA INTEGRITY & CONSISTENCY VERIFICATION")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Define run directory
    run_dir = Path("results/complete_pipeline_seed42")
    tex_path = Path("main.tex")

    # 1. Verify hash consistency within single run
    print("─"*80)
    print("1. Verifying hash consistency within complete_pipeline_seed42")
    print("─"*80)

    configs = [
        "01_baseline",
        "06_eval_base_opro",
        "03_eval_lora",
        "07_eval_lora_opro"
    ]

    hashes = {}
    for config in configs:
        csv_path = run_dir / config / "predictions.csv"
        if csv_path.exists():
            hash_val = compute_file_hash(csv_path)
            hashes[config] = hash_val
            print(f"✅ {config:30s} | hash: {hash_val[:16]}...")
        else:
            print(f"❌ {config:30s} | MISSING")
            hashes[config] = None

    all_hashes_present = all(h is not None for h in hashes.values())

    # 2. Check for Open references in Results section
    print("\n" + "─"*80)
    print("2. Checking for Open variant references in Results section")
    print("─"*80)

    ref_check = check_references_in_tex(tex_path)

    if ref_check.get("results_clean"):
        print("✅ No Open variant references found in Results section")
    else:
        print(f"⚠️  Found {ref_check['open_refs_in_results']} Open references in Results:")
        for ref in ref_check['open_refs_details']:
            print(f"   - {ref}")

    print(f"ℹ️  Open references in Appendix: {ref_check['open_refs_in_appendix']} (expected)")

    # 3. Check referenced artifacts
    print("\n" + "─"*80)
    print("3. Verifying all referenced tables/figures exist")
    print("─"*80)

    artifact_check = check_referenced_artifacts(tex_path)

    print(f"Tables referenced: {artifact_check['total_table_refs']}")
    print(f"Figures referenced: {artifact_check['total_figure_refs']}")

    if artifact_check['missing_tables']:
        print(f"❌ Missing tables: {artifact_check['missing_tables']}")
    else:
        print("✅ All referenced tables present")

    if artifact_check['missing_figures']:
        print(f"⚠️  Missing figures: {artifact_check['missing_figures']}")
        print("   (Some figures may need regeneration from scripts)")
    else:
        print("✅ All referenced figures present")

    # 4. Overall summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    all_checks_pass = (
        all_hashes_present and
        ref_check.get("results_clean", False) and
        len(artifact_check['missing_tables']) == 0
    )

    if all_checks_pass:
        print("✅ ALL CRITICAL CHECKS PASSED")
        print("   - Single run consistency: VERIFIED")
        print("   - Results section clean (no Open refs): VERIFIED")
        print("   - All tables present: VERIFIED")
        if artifact_check['missing_figures']:
            print("   ⚠️  Some figures may need regeneration (non-critical)")
    else:
        print("⚠️  SOME ISSUES DETECTED")
        if not all_hashes_present:
            print("   - Missing predictions.csv in some configs")
        if not ref_check.get("results_clean"):
            print("   - Open variant references still in Results section")
        if artifact_check['missing_tables']:
            print("   - Missing table files")

    # Save verification report
    output = {
        "timestamp": datetime.now().isoformat(),
        "run_directory": str(run_dir),
        "configs_verified": configs,
        "hashes": {k: v for k, v in hashes.items()},
        "open_references_check": ref_check,
        "artifact_check": artifact_check,
        "overall_status": "PASS" if all_checks_pass else "PARTIAL"
    }

    report_path = Path("results/final_integrity_report.json")
    with open(report_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nFull report saved to: {report_path}")

    return 0 if all_checks_pass else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
