#!/bin/bash
# ============================================================
# run_sequential_gates_GitHub.sh
# Runs every gating step after unmixing in one go, then makes the
# per-well gating QC PDF (workflow.txt Steps 7 and 8).
#
# Gating hierarchy:
#   Gate 1: FSC vs SSC      → already done before unmixing (workflow Step 3)
#   Gate 2: Viability       → viability-dye threshold (keep live)
#   Gate 3: CD11c+          → CD11c threshold (keep positive)
#   Gate 4a: BMDCs          → MHC-II hi CD11b int polygon
#   Gate 4b: BMDMs          → MHC-II int CD11b hi polygon
#
# Usage:
#   1. Edit the "EDIT FOR YOUR DATA" block below.
#   2. Run from your project folder (relative paths start there):
#        bash path/to/run_sequential_gates_GitHub.sh
#   The Python scripts are found next to this file, wherever you run it from.
# ============================================================

set -e

# ═════════════════════════════════════════════════════════════
# EDIT FOR YOUR DATA
# ═════════════════════════════════════════════════════════════

# Unmixed .fcs files to gate (output of batch_unmix_GitHub.py)
INPUT_DIR="results/unmixed/Exp1"

# Gate files drawn in Flow_View_GitHub.py
GATES="analysis/Exp1/gates"
VIABILITY_THRESHOLD="$GATES/LD_gate_threshold.json"
CD11C_THRESHOLD="$GATES/CD11cpos_gate_threshold.json"
BMDCS_HULL="$GATES/BMDCs_gate_hull.txt"
BMDMS_HULL="$GATES/BMDMs_gate_hull.txt"

# FCS channel names in your panel. The polygon hulls must use the same
# x/y axes they were drawn on in Flow_View (here x = MHC-II, y = CD11b).
CD11C_CHANNEL="BUV395"
MHCII_CHANNEL="PERCP"
CD11B_CHANNEL="APC"

# Output folders (one per gating step) and the QC PDF
OUT_LIVE="analysis/Exp1/gated/Exp1_live"
OUT_CD11C="analysis/Exp1/gated/Exp1_cd11cpos"
OUT_BMDCS="analysis/Exp1/gated/Exp1_BMDCs"
OUT_BMDMS="analysis/Exp1/gated/Exp1_BMDMs"
QC_PDF="results/plots/Exp1_gating_QC.pdf"

# Python to use (e.g. venv/bin/python)
PYTHON="python"

# ═════════════════════════════════════════════════════════════
# END OF SETTINGS
# ═════════════════════════════════════════════════════════════

SCRIPTS="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "============================================================"
echo " Sequential gating pipeline"
echo "============================================================"
echo " Input : $INPUT_DIR"
echo " Gates : $GATES"
echo ""

# ── Check inputs exist ────────────────────────────────────────
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌  Input folder not found: $INPUT_DIR"
    exit 1
fi
for f in "$VIABILITY_THRESHOLD" "$CD11C_THRESHOLD" "$BMDCS_HULL" "$BMDMS_HULL"; do
    if [ ! -f "$f" ]; then
        echo "❌  Missing gate file: $f"
        exit 1
    fi
done
echo "✓ Input folder and all gate files found"
echo ""

# ── GATE 2: Viability ─────────────────────────────────────────
echo "------------------------------------------------------------"
echo " GATE 2/4 — Viability (keep live)"
echo "------------------------------------------------------------"
"$PYTHON" "$SCRIPTS/Apply_sequential_gates_GitHub.py" \
    --input     "$INPUT_DIR" \
    --output    "$OUT_LIVE" \
    --threshold "$VIABILITY_THRESHOLD"
echo ""

# ── GATE 3: CD11c+ ───────────────────────────────────────────
echo "------------------------------------------------------------"
echo " GATE 3/4 — CD11c+"
echo "------------------------------------------------------------"
"$PYTHON" "$SCRIPTS/Apply_sequential_gates_GitHub.py" \
    --input     "$OUT_LIVE" \
    --output    "$OUT_CD11C" \
    --threshold "$CD11C_THRESHOLD"
echo ""

# ── GATE 4a: BMDCs ───────────────────────────────────────────
echo "------------------------------------------------------------"
echo " GATE 4a/4 — BMDCs (MHC-II hi CD11b int)"
echo "------------------------------------------------------------"
"$PYTHON" "$SCRIPTS/Apply_sequential_gates_GitHub.py" \
    --input   "$OUT_CD11C" \
    --output  "$OUT_BMDCS" \
    --hull    "$BMDCS_HULL" \
    --hull-x  "$MHCII_CHANNEL" \
    --hull-y  "$CD11B_CHANNEL"
echo ""

# ── GATE 4b: BMDMs ───────────────────────────────────────────
echo "------------------------------------------------------------"
echo " GATE 4b/4 — BMDMs (MHC-II int CD11b hi)"
echo "------------------------------------------------------------"
"$PYTHON" "$SCRIPTS/Apply_sequential_gates_GitHub.py" \
    --input   "$OUT_CD11C" \
    --output  "$OUT_BMDMS" \
    --hull    "$BMDMS_HULL" \
    --hull-x  "$MHCII_CHANNEL" \
    --hull-y  "$CD11B_CHANNEL"
echo ""

# ── Gating QC PDF ─────────────────────────────────────────────
echo "------------------------------------------------------------"
echo " Generating per-well gating QC PDF"
echo "------------------------------------------------------------"
"$PYTHON" "$SCRIPTS/make_gate_qc_pdf_GitHub.py" \
    --pre-gate        "$INPUT_DIR" \
    --post-live       "$OUT_LIVE" \
    --post-cd11c      "$OUT_CD11C" \
    --post-bmdcs      "$OUT_BMDCS" \
    --post-bmdms      "$OUT_BMDMS" \
    --ld-threshold    "$VIABILITY_THRESHOLD" \
    --cd11c-threshold "$CD11C_THRESHOLD" \
    --bmdcs-hull      "$BMDCS_HULL" \
    --bmdms-hull      "$BMDMS_HULL" \
    --cd11c-channel   "$CD11C_CHANNEL" \
    --mhcii-channel   "$MHCII_CHANNEL" \
    --out             "$QC_PDF"
echo ""

echo "============================================================"
echo " Sequential gating complete."
echo ""
echo " Populations:"
echo "   Live            : $OUT_LIVE"
echo "   All live CD11c+ : $OUT_CD11C"
echo "   BMDCs           : $OUT_BMDCS"
echo "   BMDMs           : $OUT_BMDMS"
echo " QC PDF            : $QC_PDF"
echo "============================================================"
