#!/usr/bin/env python3
"""
figures_BMDC_activation_v2_GitHub.py
Dendritic-cell (BMDC) activation figure, pooled across experiments/mice,
at one lipid A dose, with significance stars against BOTH the vehicle
control and the positive control on the same panel.

>>> To use with your own data, edit only the USER SETTINGS block below. <<<
Everything after that block reads those settings; no other paths are set
anywhere in the script.

Plot: one panel per activation marker (CD40, CD80, CD86, PDL1), stacked.
    X = strain × Rf band. Y = arcsinh MFI shift from vehicle ctrl (DMSO).
    Each point = one well. Color = strain. Shape = mouse.
    Bar = median per strain × Rf. Red dashed = positive-control reference
    (E. coli lipid A, labelled "E. coli_Sigma").
    Grey/black stars = sig. vs vehicle ctrl. Red stars = sig. vs positive ctrl.
    Legend is written on a separate second PDF page.

Calculation (per well): arcsinh(cofactor 150) is applied to every cell,
the arcsinh of the pooled vehicle-ctrl median is subtracted, and the median
over cells is taken. The bar is the median of those per-well values.
Stats (default "vs_both" mode): for each strain × Rf group (n >= 2 wells),
a one-sample t-test of the per-well values against 0 (= vehicle ctrl) and a
second against the positive-control reference value. No multiple-comparison
correction. Stars: * P<0.10, ** P<0.05, *** P<0.01, **** P<0.001.
(Other modes exist in the code: "vs_ctrl", "vs_pos_ctrl", and
"within_strain" = one-way ANOVA across Rf bands (gate P<0.10) then pairwise
Welch t-tests, drawn as brackets.)

The positive-control reference is DOSE-MATCHED: only positive_ctrl wells at
the same dose_ng as the plotted lipid A wells are used. Mice with no lipid A
wells at the chosen dose simply contribute nothing (e.g. Exp2 only has
1000 ng/mL, so it drops out at --dose 100).

Usage:
    python figures_BMDC_activation_v2_GitHub.py --dose 1000 \\
        --population BMDCs --out results/plots/BMDC_activation_1000ng.pdf
    python figures_BMDC_activation_v2_GitHub.py --dose 100 \\
        --population BMDCs --out results/plots/BMDC_activation_100ng.pdf
    (--population also accepts cd11cpos or BMDMs; --base-dir sets the folder
     that relative paths in EXPERIMENTS start from, default = current folder)
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════
# USER SETTINGS — EDIT THIS BLOCK FOR YOUR DATA
# ═════════════════════════════════════════════════════════════

# ── 1. Your experiments: one entry per mouse (or per plate) ──
#
#   mouse_id        short ID used internally (any unique text)
#   legend          name shown in the figure legend (legend is sorted by it)
#   point_shape     matplotlib marker for this mouse's points: o s ^ D v P X
#   gated_dir       folder of gated .fcs files. "{population}" is replaced by
#                   --population (BMDCs / cd11cpos / BMDMs). Relative paths
#                   start from --base-dir; absolute paths are used as-is.
#   plate_map       plate-map CSV for this mouse (columns described below)
#   mouse_filter    if one plate map holds several mice, the value in its
#                   "mouse" column for this mouse; otherwise None
#   failed_markers  markers whose antibody failed for this mouse; they are
#                   left out of that marker's panel (e.g. ["CD80", "CD86"])
#   rf_snap         optional table snapping slightly different Rf labels to
#                   the CANONICAL_RF values below; otherwise None
#
# Plate-map CSV columns (one row per well):
#   well, mouse, condition, strain, rf_band, dose_ng, group, ignore, notes
#   group  = lipid_A | vehicle_ctrl | antigen_ctrl | positive_ctrl
#   ignore = 1 drops the well (e.g. clogged), else 0
#   "mouse" is only needed when mouse_filter is used.
# Gated .fcs file names must begin with the well ID and a space
# ("A8 anything.fcs") so they can be matched to the plate map.
#
# The entries below are the original experiments, kept as a worked example:
# Exp2 = mouse 9, Exp3 = mouse 8 (CD80/CD86 antibodies failed),
# Exp4 = mice 7 and 10 sharing one plate map.

# Exp2's Rf labels were written slightly differently (0.68 vs 0.684, ...)
EXP2_RF_SNAP = {
    "381":  {0.68: 0.684, 0.78: 0.789, 0.85: 0.857},
    "1587": {0.68: 0.684, 0.80: 0.805, 0.86: 0.868},
}

EXPERIMENTS = [
    dict(mouse_id="Exp2_M9",  legend="C57/BL6-2", point_shape="o",
         gated_dir="analysis/Exp2/gated/Exp2_M9_{population}",
         plate_map="analysis/Exp2/Exp2_map.csv",
         mouse_filter=None, failed_markers=[], rf_snap=EXP2_RF_SNAP),
    dict(mouse_id="Exp3_M8",  legend="C57/BL6-3", point_shape="s",
         gated_dir="analysis/Exp3/gated/Exp3_M8_{population}",
         plate_map="analysis/Exp3/Exp3_map.csv",
         mouse_filter=None, failed_markers=["CD80", "CD86"], rf_snap=None),
    dict(mouse_id="Exp4_M7",  legend="C57/BL6-1", point_shape="^",
         gated_dir="analysis/Exp4/gated/Exp4_M7_{population}",
         plate_map="analysis/Exp4/Exp4_map.csv",
         mouse_filter="M7",  failed_markers=[], rf_snap=None),
    dict(mouse_id="Exp4_M10", legend="C57/BL6-4", point_shape="D",
         gated_dir="analysis/Exp4/gated/Exp4_M10_{population}",
         plate_map="analysis/Exp4/Exp4_map.csv",
         mouse_filter="M10", failed_markers=[], rf_snap=None),
]

# ── 2. Your antibody panel: fluor (FCS channel name) -> marker ──
FLUOR_TO_MARKER = {
    "FITC":  "CD40",
    "PE":    "CD80",
    "BV421": "CD86",
    "BV711": "PDL1",
}
ACTIVATION_MARKERS = ["CD40", "CD80", "CD86", "PDL1"]   # one panel each, top to bottom
MARKER_COLORS = {
    "CD40": "#1f78b4",
    "CD80": "#e31a1c",
    "CD86": "#ff7f00",
    "PDL1": "#33a02c",
}

# ── 3. Your lipid A preparations: strains and their Rf bands ──
# "strain" and "rf_band" must match the plate-map columns.
STRAIN_ORDER  = ["381", "1587", "1773", "EC_LPXE"]
STRAIN_COLORS = {
    "381":     "#2166ac",
    "1587":    "#d6604d",
    "1773":    "#4dac26",
    "EC_LPXE": "#8B008B",
}
STRAIN_LABELS = {                       # full names, legend page
    "381":     "381 WT",
    "1587":    "1587 C4′-phos mut",
    "1773":    "1773 C1-phos mut",
    "EC_LPXE": "E. coli + LpxE",
}
STRAIN_LABELS_SHORT = {                 # compact names, top-panel header
    "381":     "381",
    "1587":    "1587",
    "1773":    "1773",
    "EC_LPXE": "EC+LpxE",
}
CANONICAL_RF = {                        # x-axis; wells with other Rf values are skipped
    "381":     [0.684, 0.789, 0.857],
    "1587":    [0.684, 0.805, 0.868],
    "1773":    [0.83],
    "EC_LPXE": [0.46, 0.67, 0.78, 0.82],
}

# ═════════════════════════════════════════════════════════════
# END OF USER SETTINGS — nothing below needs editing
# ═════════════════════════════════════════════════════════════

MOUSE_MARKERS       = {e["mouse_id"]: e["point_shape"] for e in EXPERIMENTS}
MOUSE_DISPLAY_NAMES = {e["mouse_id"]: e["legend"] for e in EXPERIMENTS}
MOUSE_LEGEND_ORDER  = sorted(MOUSE_DISPLAY_NAMES, key=MOUSE_DISPLAY_NAMES.get)


# ─────────────────────────────────────────────────────────────
# FCS READER
# ─────────────────────────────────────────────────────────────

def _parse_text(text_block):
    if not text_block:
        return {}
    delim = text_block[0]
    ph = "\x00"
    text_block = text_block.replace(delim * 2, ph)
    parts = [p.replace(ph, delim) for p in text_block.strip(delim).split(delim)]
    d = {}
    for i in range(0, len(parts) - 1, 2):
        d[parts[i].strip().lower()] = parts[i + 1].strip()
    return d


def read_fcs(path):
    with open(path, "rb") as f:
        raw = f.read()
    hdr = raw[:58].decode("ascii", errors="ignore")
    ts, te = int(hdr[10:18].strip() or 0), int(hdr[18:26].strip() or 0)
    ds, de = int(hdr[26:34].strip() or 0), int(hdr[34:42].strip() or 0)
    t = _parse_text(raw[ts:te + 1].decode("ascii", errors="ignore"))

    def meta(k, d=None):
        return t.get(k, t.get(f"${k}", d))

    ds  = int(meta("begindata", ds))
    de  = int(meta("enddata",   de))
    par = int(meta("par", 0))
    tot = int(meta("tot", 0))
    raw_data = raw[ds:de + 1]
    dtype  = str(meta("datatype", "F")).upper()
    little = str(meta("byteord", "1,2,3,4")).startswith("1")

    if dtype == "F":
        dt = np.dtype("<f4" if little else ">f4")
    elif dtype == "D":
        dt = np.dtype("<f8" if little else ">f8")
    else:
        bits = int(meta("p1b", 32))
        dt = np.dtype(f"{'<' if little else '>'}u{bits // 8}")

    arr   = np.frombuffer(raw_data, dtype=dt)
    names = []
    for i in range(1, par + 1):
        cands = [meta(f"p{i}n"), meta(f"p{i}s")]
        names.append(next((c for c in cands if c and c.strip()), f"Ch{i}"))

    n = tot if tot > 0 and len(arr) == tot * par else len(arr) // par
    return pd.DataFrame(arr[:n * par].reshape((n, par)), columns=names)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _arcsinh(x, cofactor=150):
    return np.arcsinh(np.asarray(x, dtype=float) / cofactor)


def snap_rf(rf_val, strain, snap_table=None):
    try:
        rf = float(rf_val)
    except (ValueError, TypeError):
        return rf_val
    if snap_table and strain in snap_table:
        return snap_table[strain].get(rf, rf)
    return rf


def _baseline_medians(records, group):
    """Pool all wells of a given group and return per-marker medians."""
    cells = [r["_df"] for r in records if r["group"] == group]
    if not cells:
        return {}
    pooled = pd.concat(cells, ignore_index=True)
    return {m: float(np.median(pooled[m].dropna().values))
            for m in ACTIVATION_MARKERS if m in pooled.columns}


def _well_shifts(records, baselines, mouse_id, dose_ng=1000):
    """
    For each lipid_A well at dose_ng in canonical Rf set,
    compute per-marker median arcsinh shift from baselines.
    """
    rows = []
    for r in records:
        if r["group"] != "lipid_A" or r["dose_ng"] != dose_ng:
            continue
        if r["strain"] not in CANONICAL_RF:
            continue
        try:
            rf = float(r["rf_band"])
        except (ValueError, TypeError):
            continue
        if rf not in CANONICAL_RF.get(r["strain"], []):
            continue
        row_dict = {"mouse": mouse_id, "strain": r["strain"],
                    "rf_band": rf, "well": r["well"]}
        df = r["_df"]
        for marker, base in baselines.items():
            if marker in df.columns:
                vals = df[marker].dropna().values
                row_dict[marker] = float(np.median(_arcsinh(vals) - _arcsinh(base)))
            else:
                row_dict[marker] = np.nan
        rows.append(row_dict)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _ecstore_shifts(records, baselines):
    """
    Average arcsinh shift of all positive_ctrl (EC.store) wells
    vs the given baselines. Returns per-marker float dict.
    """
    ec_cells = [r["_df"] for r in records if r["group"] == "positive_ctrl"]
    if not ec_cells or not baselines:
        return {}
    pooled = pd.concat(ec_cells, ignore_index=True)
    result = {}
    for marker, base in baselines.items():
        if marker in pooled.columns:
            vals = pooled[marker].dropna().values
            result[marker] = float(np.median(_arcsinh(vals) - _arcsinh(base)))
    return result


# ─────────────────────────────────────────────────────────────
# LOAD ONE MOUSE
# ─────────────────────────────────────────────────────────────

def load_mouse(bmdcs_dir, map_file, mouse_id,
               mouse_filter=None, rf_snap=None, fluor_map=None, dose_ng=1000):
    """
    Returns dict:
        wells_dmso  — per-well shifts vs vehicle_ctrl
        wells_ova   — per-well shifts vs antigen_ctrl (empty if none)
        ec_dmso     — {marker: float} EC.store shift vs vehicle_ctrl
        ec_ova      — {marker: float} EC.store shift vs antigen_ctrl

    EC.store (positive_ctrl) wells are dose-matched to dose_ng — the plate
    maps tag EC.store wells with both 100ng and 1000ng dose_ng, and mixing
    them would blend two different positive-control potencies into one
    reference value.
    """
    bmdcs_dir = Path(bmdcs_dir)
    plate_map = pd.read_csv(map_file)

    if mouse_filter is not None:
        plate_map = plate_map[plate_map["mouse"] == mouse_filter].copy()

    if rf_snap:
        plate_map["rf_band"] = plate_map.apply(
            lambda r: snap_rf(r["rf_band"], str(r["strain"]), rf_snap), axis=1)

    fcs_files = [f for f in sorted(bmdcs_dir.glob("*.fcs"))
                 if f.parent == bmdcs_dir]
    records = []
    for fcs_path in fcs_files:
        well = fcs_path.name.split(" ")[0]
        row  = plate_map[plate_map["well"] == well]
        if row.empty:
            continue
        row = row.iloc[0]
        if row.get("ignore", 0) == 1:
            continue
        try:
            df = read_fcs(str(fcs_path))
        except Exception as e:
            print(f"    ✗ {fcs_path.name}: {e}")
            continue
        if len(df) == 0:
            continue

        active_map = fluor_map if fluor_map is not None else FLUOR_TO_MARKER
        rename = {}
        for col in df.columns:
            for fluor, marker in active_map.items():
                if col.upper().strip() == fluor.upper():
                    rename[col] = marker
                    break
        df = df.rename(columns=rename)

        records.append({
            "well":    well,
            "group":   row["group"],
            "strain":  str(row["strain"]),
            "rf_band": row["rf_band"],
            "dose_ng": row["dose_ng"],
            "_df":     df,
        })

    if not records:
        print(f"  ⚠ No records from {bmdcs_dir}")
        return {"wells_dmso": pd.DataFrame(), "wells_ova": pd.DataFrame(),
                "ec_dmso": {}, "ec_ova": {}}

    dmso_base = _baseline_medians(records, "vehicle_ctrl")
    ova_base  = _baseline_medians(records, "antigen_ctrl")

    if not dmso_base:
        print(f"  ⚠ No vehicle_ctrl for {mouse_id}")
        return {"wells_dmso": pd.DataFrame(), "wells_ova": pd.DataFrame(),
                "ec_dmso": {}, "ec_ova": {}}

    wells_dmso = _well_shifts(records, dmso_base, mouse_id, dose_ng=dose_ng)
    wells_ova  = _well_shifts(records, ova_base,  mouse_id, dose_ng=dose_ng) if ova_base else pd.DataFrame()

    ec_records = [r for r in records
                  if not (r["group"] == "positive_ctrl" and r["dose_ng"] != dose_ng)]
    ec_dmso    = _ecstore_shifts(ec_records, dmso_base)
    ec_ova     = _ecstore_shifts(ec_records, ova_base) if ova_base else {}

    markers_found = [m for m in ACTIVATION_MARKERS if m in dmso_base]
    has_ova = "yes" if ova_base else "no"
    has_ec  = "yes" if ec_dmso  else "no"
    print(f"    {mouse_id}: {len(wells_dmso)} wells | "
          f"markers: {', '.join(markers_found)} | OVA: {has_ova} | EC.store: {has_ec}")

    return {"wells_dmso": wells_dmso, "wells_ova": wells_ova,
            "ec_dmso": ec_dmso, "ec_ova": ec_ova}


# ─────────────────────────────────────────────────────────────
# LOAD ALL EXPERIMENTS
# (reads the EXPERIMENTS list from USER SETTINGS)
# ─────────────────────────────────────────────────────────────

def _experiment_paths(base_dir, population):
    """Resolve each EXPERIMENTS entry's gated_dir / plate_map. Stops with a
    clear message listing every missing path, so a typo in USER SETTINGS
    can't silently drop a mouse from the figure."""
    paths, missing = [], []
    for e in EXPERIMENTS:
        gated = base_dir / e["gated_dir"].format(population=population)
        pmap  = base_dir / e["plate_map"]
        for label, p, is_dir in (("gated_dir", gated, True), ("plate_map", pmap, False)):
            if not (p.is_dir() if is_dir else p.is_file()):
                missing.append(f"  {e['mouse_id']}  {label}: {p}")
        paths.append((gated, pmap))
    if missing:
        raise SystemExit(
            "These paths from EXPERIMENTS (USER SETTINGS at the top of the "
            "script) were not found:\n" + "\n".join(missing) +
            f"\nRelative paths start from --base-dir ({base_dir.resolve()}).")
    return paths


def load_all(base_dir, population="BMDCs", dose_ng=1000):
    base_dir = Path(base_dir)
    paths = _experiment_paths(base_dir, population)
    print(f"Loading {population} data at {dose_ng} ng/mL...")

    results = []
    for e, (gated, pmap) in zip(EXPERIMENTS, paths):
        r = load_mouse(gated, pmap, e["mouse_id"],
                       mouse_filter=e["mouse_filter"], rf_snap=e["rf_snap"],
                       dose_ng=dose_ng)
        # Blank out markers whose antibody failed for this mouse
        for df_key in ("wells_dmso", "wells_ova"):
            if not r[df_key].empty:
                for m in e["failed_markers"]:
                    r[df_key][m] = np.nan
        for ec_key in ("ec_dmso", "ec_ova"):
            for m in e["failed_markers"]:
                r[ec_key].pop(m, None)
        results.append(r)

    # Combine well DataFrames
    def _concat(key):
        frames = [r[key] for r in results if not r[key].empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    data_dmso = _concat("wells_dmso")
    data_ova  = _concat("wells_ova")

    # Global (median) EC.store shift across mice that have them — matches
    # how the experimental groups' global MFI is aggregated across
    # replicates (median, not mean), so the positive-control reference
    # value used in the vs-positive-ctrl t-test is computed the same way
    # as the values it's being compared against.
    def _avg_ec(key):
        per_marker = {m: [] for m in ACTIVATION_MARKERS}
        for r in results:
            for m, v in r[key].items():
                if not np.isnan(v):
                    per_marker[m].append(v)
        return {m: float(np.median(vs)) for m, vs in per_marker.items() if vs}

    ec_dmso = _avg_ec("ec_dmso")
    ec_ova  = _avg_ec("ec_ova")

    print(f"  DMSO wells: {len(data_dmso)}  |  OVA wells: {len(data_ova)}")
    return data_dmso, data_ova, ec_dmso, ec_ova


# ─────────────────────────────────────────────────────────────
# BUILD X-AXIS
# ─────────────────────────────────────────────────────────────

def build_xaxis():
    positions, xtick_pos, xtick_lab, spans = {}, [], [], []
    x = 0
    for strain in STRAIN_ORDER:
        x_start = x
        for rf in CANONICAL_RF[strain]:
            positions[(strain, rf)] = x
            xtick_pos.append(x)
            xtick_lab.append(f"{rf:.2f}")
            x += 1
        spans.append((strain, x_start, x - 1))
    return positions, xtick_pos, xtick_lab, spans


# ─────────────────────────────────────────────────────────────
# STATISTICS HELPERS
# ─────────────────────────────────────────────────────────────

def _pval_label(p):
    if p < 0.001: return "****"
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return "ns"


def _add_stats_to_panel(ax, mdf, marker, positions, spans, stats_mode, ec_val=None):
    sub = mdf[mdf[marker].notna()] if not mdf.empty else mdf
    if sub.empty:
        return
    y_vals  = sub[marker].values
    y_range = float(max(y_vals) - min(y_vals)) or 1.0
    y_max   = float(max(y_vals))

    if stats_mode == "vs_pos_ctrl":
        if ec_val is None or np.isnan(ec_val):
            return
        for (strain, rf_band), grp in sub.groupby(["strain", "rf_band"]):
            vals = grp[marker].dropna().values
            if len(vals) < 2:
                continue
            _, p = scipy_stats.ttest_1samp(vals, ec_val)
            lbl  = _pval_label(p)
            key  = (strain, rf_band)
            if key not in positions:
                continue
            x   = positions[key]
            med = float(np.median(vals))
            # Place label between median and EC.store line for readability
            label_y = med + 0.07 * abs(ec_val - med) if ec_val != med else med + 0.07 * y_range
            ax.text(x, label_y, lbl,
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="#c00000" if lbl != "ns" else "#aaaaaa", zorder=7)

    elif stats_mode == "vs_ctrl":
        for (strain, rf_band), grp in sub.groupby(["strain", "rf_band"]):
            vals = grp[marker].dropna().values
            if len(vals) < 2:
                continue
            _, p = scipy_stats.ttest_1samp(vals, 0)
            lbl  = _pval_label(p)
            key  = (strain, rf_band)
            if key not in positions:
                continue
            x   = positions[key]
            med = float(np.median(vals))
            ax.text(x, med + 0.07 * y_range, lbl,
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="#c00000" if lbl != "ns" else "#aaaaaa", zorder=7)

    elif stats_mode == "vs_both":
        # Fixed rows near the top of the panel (x = data coord, y = axes fraction)
        # rather than tied to each group's data height. Add headroom above the
        # data so tall points don't collide with the star rows.
        cur_bottom, cur_top = ax.get_ylim()
        ax.set_ylim(top=cur_top + 0.30 * (cur_top - cur_bottom))
        xaxis_t = ax.get_xaxis_transform()
        y_veh_row = 0.90
        y_pos_row = 0.97
        for (strain, rf_band), grp in sub.groupby(["strain", "rf_band"]):
            vals = grp[marker].dropna().values
            if len(vals) < 2:
                continue
            key = (strain, rf_band)
            if key not in positions:
                continue
            x = positions[key]

            # vs vehicle control (one-sample t-test vs 0)
            _, p_veh = scipy_stats.ttest_1samp(vals, 0)
            lbl_veh  = _pval_label(p_veh)
            ax.text(x, y_veh_row, lbl_veh, transform=xaxis_t,
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="#333333" if lbl_veh != "ns" else "#aaaaaa",
                    zorder=7, clip_on=False)

            # vs positive control (one-sample t-test vs EC.store)
            if ec_val is not None and not np.isnan(ec_val):
                _, p_pos = scipy_stats.ttest_1samp(vals, ec_val)
                lbl_pos  = _pval_label(p_pos)
                ax.text(x, y_pos_row, lbl_pos, transform=xaxis_t,
                        ha="center", va="center", fontsize=8, fontweight="bold",
                        color="#c00000" if lbl_pos != "ns" else "#e6b8b8",
                        zorder=7, clip_on=False)

    elif stats_mode == "within_strain":
        bracket_top  = y_max + 0.18 * y_range
        bracket_step = 0.12 * y_range
        max_top      = bracket_top

        for strain, _, _ in spans:
            rf_bands = CANONICAL_RF.get(strain, [])
            if len(rf_bands) < 2:
                continue
            strain_groups = {}
            for rf in rf_bands:
                grp  = sub[(sub["strain"] == strain) & (sub["rf_band"] == rf)]
                vals = grp[marker].dropna().values
                if len(vals) >= 2:
                    strain_groups[rf] = vals
            if len(strain_groups) < 2:
                continue
            try:
                _, p_anova = scipy_stats.f_oneway(*strain_groups.values())
            except Exception:
                continue
            if p_anova >= 0.10:
                continue
            rf_list  = sorted(strain_groups.keys())
            drew_any = False
            for i, rf1 in enumerate(rf_list):
                for rf2 in rf_list[i + 1:]:
                    try:
                        _, p_pair = scipy_stats.ttest_ind(
                            strain_groups[rf1], strain_groups[rf2], equal_var=False)
                    except Exception:
                        continue
                    lbl = _pval_label(p_pair)
                    if lbl == "ns":
                        continue
                    x1 = positions.get((strain, rf1))
                    x2 = positions.get((strain, rf2))
                    if x1 is None or x2 is None:
                        continue
                    tick = bracket_step * 0.18
                    ax.plot([x1, x1, x2, x2],
                            [bracket_top - tick, bracket_top,
                             bracket_top, bracket_top - tick],
                            color="#444", lw=0.9, zorder=6, clip_on=False)
                    ax.text((x1 + x2) / 2, bracket_top + tick * 0.4, lbl,
                            ha="center", va="bottom", fontsize=9,
                            fontweight="bold", color="#c00000", clip_on=False)
                    max_top      = max(max_top, bracket_top + bracket_step)
                    bracket_top += bracket_step
                    drew_any     = True
            if drew_any:
                bracket_top += bracket_step * 0.5

        ax.set_ylim(top=max_top + bracket_step * 0.5)


# ─────────────────────────────────────────────────────────────
# PAGE GEOMETRY — 8.5x11 letter page, 1-inch margins
# ─────────────────────────────────────────────────────────────

PAGE_W_IN, PAGE_H_IN = 8.5, 11.0
MARGIN_IN = 1.0


def _content_box():
    """Figure-fraction (left, right, bottom, top) of the printable area
    inside a 1-inch margin on an 8.5x11 page."""
    left   = MARGIN_IN / PAGE_W_IN
    right  = 1 - MARGIN_IN / PAGE_W_IN
    bottom = MARGIN_IN / PAGE_H_IN
    top    = 1 - MARGIN_IN / PAGE_H_IN
    return left, right, bottom, top


# ─────────────────────────────────────────────────────────────
# DRAW DATA PAGE — 4 stacked panels, one per marker
# ─────────────────────────────────────────────────────────────

def _draw_data_page(pdf, data, ec_refs,
                    baseline_label, stats_mode=None):
    positions, xtick_pos, xtick_lab, spans = build_xaxis()

    left, right, bottom, top = _content_box()
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))

    # Reserved bands (inches -> figure-fraction) so nothing drawn outside the
    # axes proper bleeds past the 1-inch page margin. No title/subtitle band —
    # this figure is embedded under an external caption, so every bit of
    # vertical room goes to the panels themselves.
    top_label_h = 0.45 / PAGE_H_IN   # top: room for strain labels (shifted up to clear in-plot stats)
    label_col_w = 1.15 / PAGE_W_IN   # left: marker name + shared ylabel + y-ticks
    right_col_w = 1.1  / PAGE_W_IN   # right: "E. coli_Sigma" annotation + N count
    xlabel_h    = 0.72 / PAGE_H_IN   # bottom: x-ticks + "Rf band" label (bottom panel only)

    gs = fig.add_gridspec(4, 1,
                          left=left + label_col_w, right=right - right_col_w,
                          top=top - top_label_h, bottom=bottom + xlabel_h,
                          hspace=0.18)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]

    # Shared y-axis label spanning all four panels, centered vertically —
    # sits closer to the axes; marker name (CD40 etc.) sits further out.
    fig.text(left + 0.50 / PAGE_W_IN, (top - top_label_h + bottom + xlabel_h) / 2,
             f"arcsinh shift from {baseline_label}",
             rotation=90, ha="center", va="center", fontsize=15)

    np.random.seed(42)
    n_panels = len(ACTIVATION_MARKERS)
    for i, (ax, marker) in enumerate(zip(axes, ACTIVATION_MARKERS)):
        is_top    = (i == 0)
        is_bottom = (i == n_panels - 1)
        # Mice whose antibody failed for this marker are already NaN (failed_markers)
        mdf = data.copy() if not data.empty else pd.DataFrame()
        mdf = mdf[mdf[marker].notna()] if not mdf.empty else mdf

        # Alternating strain background; strain labels on top panel only
        for j, (strain, x0, x1) in enumerate(spans):
            ax.axvspan(x0 - 0.5, x1 + 0.5,
                       facecolor="#f0f0f0" if j % 2 == 0 else "white",
                       zorder=0, alpha=0.5)
            if is_top:
                ax.text((x0 + x1) / 2, 1.12,
                        STRAIN_LABELS_SHORT.get(strain, strain),
                        ha="center", va="bottom", fontsize=10.5,
                        color=STRAIN_COLORS[strain], fontweight="bold",
                        transform=ax.get_xaxis_transform(), clip_on=False)

        if not mdf.empty:
            for (strain, rf_band), grp in mdf.groupby(["strain", "rf_band"]):
                key = (strain, rf_band)
                if key not in positions:
                    continue
                x_base = positions[key]
                color  = STRAIN_COLORS.get(strain, "#888")

                for _, r in grp.iterrows():
                    val = r[marker]
                    if np.isnan(val):
                        continue
                    mk = MOUSE_MARKERS.get(r["mouse"], "o")
                    ax.scatter(x_base + np.random.uniform(-0.15, 0.15), val,
                               color=color, marker=mk, s=55, alpha=0.85,
                               zorder=4, edgecolors="white", linewidths=0.5)

                med = grp[marker].median()
                ax.hlines(med, x_base - 0.28, x_base + 0.28,
                          color=color, lw=2.8, zorder=5)

            # N per Rf band (take the mode count across all strain×Rf positions)
            counts = mdf.groupby(["strain", "rf_band"])[marker].count()
            n_per_band = int(counts.median()) if len(counts) else 0
            n_mice = mdf["mouse"].nunique()
            ax.text(1.01, 0.80,
                    f"N = {n_per_band} ({n_mice} mice)",
                    transform=ax.transAxes, ha="left", va="center",
                    fontsize=10.5, color="#555", clip_on=False)

        # Positive control (E. coli_Sigma) reference line
        ec_val = ec_refs.get(marker)
        if ec_val is not None and not np.isnan(ec_val):
            ax.axhline(ec_val, color="#d62728", lw=1.6, ls="--",
                       alpha=0.85, zorder=3)
            ax.text(1.01, ec_val, "E. coli_Sigma",
                    transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=11.5,
                    color="#d62728", clip_on=False)

        ax.axhline(0, color="#bbbbbb", lw=0.8, ls="--", alpha=0.6, zorder=1)
        ax.set_xticks(xtick_pos)
        if is_bottom:
            ax.set_xticklabels(xtick_lab, fontsize=12, rotation=45, ha="right")
            ax.set_xlabel("Rf band", fontsize=15)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(axis="y", labelsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for _, x0, _ in spans[1:]:
            ax.axvline(x0 - 0.5, color="#cccccc", lw=0.8, zorder=1)

        if stats_mode is not None and not mdf.empty:
            _add_stats_to_panel(ax, mdf, marker, positions, spans, stats_mode, ec_val=ec_val)

        # Marker name further out (outermost left label); shared ylabel sits closer to axes
        ax.text(-0.22, 0.5, marker, transform=ax.transAxes, rotation=90,
                ha="center", va="center", fontsize=19, fontweight="bold",
                color=MARKER_COLORS.get(marker, "#333"), clip_on=False)

    pdf.savefig(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# DRAW LEGEND PAGE — separate page, 4 rows x 3 columns
# ─────────────────────────────────────────────────────────────

def _draw_legend_page(pdf, data, baseline_label, stats_mode):
    left, right, bottom, top = _content_box()
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))

    all_mice = [m for m in MOUSE_LEGEND_ORDER
                if not data.empty and m in data["mouse"].values]
    legend_handles = (
        [mlines.Line2D([0], [0], marker=MOUSE_MARKERS.get(m, "o"),
                       color="#555", linestyle="None", markersize=13,
                       label=MOUSE_DISPLAY_NAMES.get(m, m))
         for m in all_mice]
        + [mlines.Line2D([0], [0], color=STRAIN_COLORS[s], linestyle="-",
                         linewidth=3, label=STRAIN_LABELS[s])
           for s in STRAIN_ORDER]
        + [mlines.Line2D([0], [0], color="#d62728", linestyle="--",
                         linewidth=1.6, label="E. coli_Sigma (Global MFI)")]
        + [mlines.Line2D([0], [0], color="#bbbbbb", linestyle="--",
                         linewidth=1.2, label="0 = vehicle ctrl")]
    )
    if stats_mode == "vs_both":
        legend_handles += [
            mlines.Line2D([0], [0], marker="*", color="#333333",
                          linestyle="None", markersize=15,
                          label="sig. vs vehicle ctrl"),
            mlines.Line2D([0], [0], marker="*", color="#c00000",
                          linestyle="None", markersize=15,
                          label="sig. vs positive ctrl"),
        ]

    fig.legend(handles=legend_handles, loc="center",
               ncol=3, fontsize=12.5, frameon=False,
               title="Legend", title_fontsize=17,
               columnspacing=1.1, handletextpad=0.5, labelspacing=1.1,
               bbox_to_anchor=((left + right) / 2, (bottom + top) / 2 + 0.05),
               bbox_transform=fig.transFigure)

    fig.text((left + right) / 2, (bottom + top) / 2 - 0.12,
             "* p<0.10   ** p<0.05   *** p<0.01   **** p<0.001",
             ha="center", va="center", fontsize=15)

    pdf.savefig(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BMDC activation figure at one lipid A dose")
    parser.add_argument("--out",        required=True, help="Output PDF path")
    parser.add_argument("--dose",       type=int, default=1000,
                        help="Lipid A dose in ng/mL, matching the plate-map "
                             "dose_ng column (default: 1000)")
    parser.add_argument("--base-dir",   default=".",
                        help="Folder that relative paths in EXPERIMENTS start "
                             "from (default: current folder)")
    parser.add_argument("--population", default="BMDCs",
                        choices=["BMDCs", "cd11cpos", "BMDMs"],
                        help="Gated population to analyse (default: BMDCs)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_dmso, data_ova, ec_dmso, ec_ova = load_all(
        args.base_dir, args.population, dose_ng=args.dose)

    # The 1000 ng/mL figure is the main one; other doses get the dose in the y-label
    baseline_label = "vehicle ctrl (DMSO)"
    if args.dose != 1000:
        baseline_label += f", {args.dose} ng/mL"

    print(f"\nWriting PDF → {out_path}")
    with PdfPages(str(out_path)) as pdf:

        print("  Page 1 — data panels (vs vehicle ctrl, stats vs BOTH vehicle ctrl and positive ctrl)...")
        _draw_data_page(pdf, data_dmso, ec_dmso,
                        baseline_label=baseline_label,
                        stats_mode="vs_both")

        print("  Page 2 — legend...")
        _draw_legend_page(pdf, data_dmso,
                          baseline_label=baseline_label,
                          stats_mode="vs_both")

    print(f"\nDone. PDF saved → {out_path}")


if __name__ == "__main__":
    main()
