#!/usr/bin/env python3
"""
figures_TcellProlif_GitHub.py
Figures for OT-II T-cell proliferation (CellTrace Violet dilution) data.

DDI = Density Difference Integral: for each well, the CTV histogram
(arcsinh, cofactor 150) minus the pooled vehicle-ctrl histogram; magnitude =
positive area of the difference, sign = + if the well's median CTV is below
the vehicle median (more division = proliferation). Range -1 .. +1.

Page 1  — Figure 1: DDI (signed) overview
            X = Rf band, color = strain, shape = mouse, bar = median
            Baseline = vehicle_ctrl (subtracted); antigen_ctrl and
            pos_ctrl shown as horizontal reference lines.

Pages 2+ — Figure 2: Residual histogram grids
            One page per mouse. 2×4 grid of vehicle_ctrl-subtracted
            CTV histograms.
            Row A: controls ref + 381 WT Rf bands
            Row B: 1587 Rf bands + 1773 Rf band(s)
            Grey line = OVA ref | Orange dotted = pos_ctrl ref | Y=0 = baseline

Stats (two versions of Figure 1 are drawn): one-sample t-test of each
strain x Rf group vs the antigen-ctrl (OVA) DDI; and within-strain one-way
ANOVA across Rf bands (gate P<0.10) then pairwise Welch t-tests. No
multiple-comparison correction. * P<0.10, ** P<0.05, *** P<0.01, **** P<0.001.

Usage:
    cd <your flow project root>

    python figures_TcellProlif_GitHub.py \\
        --data  analysis/Tcell/gated/DMSO_live \\
        --out   results/plots/Tcell_figures.pdf

    # Optional CTV gate (threshold JSON from Flow_View_GitHub.py):
        --ctv-gate analysis/Tcell/gates/CTV_gate_threshold.json
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# EDIT FOR YOUR DATA: CTV channel name as it appears in your FCS files,
# strain IDs/colors/labels, and mouse IDs (marker shapes).
# ─────────────────────────────────────────────────────────────
CTV_CHANNEL  = "CellTrace Violet-A"
HIST_BINS    = np.linspace(-5, 12, 300)
HIST_CENTERS = (HIST_BINS[:-1] + HIST_BINS[1:]) / 2

STRAIN_COLORS = {
    "381":     "#2166ac",
    "1587":    "#d6604d",
    "1773":    "#4dac26",
    "EC_LPXE": "#8B008B",
    "none":    "#888888",
}
STRAIN_LABELS = {
    "381":     "381 WT",
    "1587":    "1587 C4′-phos mut",
    "1773":    "1773 C1-phos mut",
    "EC_LPXE": "EC + LPXE",
    "none":    "Control",
}
STRAIN_ORDER = ["381", "1587", "1773", "EC_LPXE"]

MOUSE_MARKERS = {
    "M7": "o", "M8": "s", "M9": "^", "MX": "D",
    "M5": "v", "M6": "P",
}

# Only antigen_ctrl and pos_ctrl shown as references (vehicle_ctrl is the zero baseline)
CTRL_STYLES = {
    "antigen_ctrl":  {"color": "#aaaaaa", "ls": "-",  "lw": 1.4,
                      "label": "OVA (antigen ctrl)"},
    "positive_ctrl": {"color": "#e6820a", "ls": ":",  "lw": 1.4,
                      "label": "pos_ctrl (EC.store)"},
}

REP_STYLES = {
    "A": {"lw": 2.0, "alpha": 0.90, "ls": "-"},
    "B": {"lw": 1.6, "alpha": 0.75, "ls": "--"},
    "C": {"lw": 1.4, "alpha": 0.65, "ls": ":"},
}


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
    data   = raw[ds:de + 1]
    dtype  = str(meta("datatype", "F")).upper()
    little = str(meta("byteord", "1,2,3,4")).startswith("1")

    if dtype == "F":
        dt = np.dtype("<f4" if little else ">f4")
    elif dtype == "D":
        dt = np.dtype("<f8" if little else ">f8")
    else:
        bits = int(meta("p1b", 32))
        dt = np.dtype(f"{'<' if little else '>'}u{bits // 8}")

    arr   = np.frombuffer(data, dtype=dt)
    names = []
    for i in range(1, par + 1):
        candidates = [meta(f"p{i}n"), meta(f"p{i}s")]
        names.append(next((c for c in candidates if c and c.strip()), f"Ch{i}"))

    n = tot if tot > 0 and len(arr) == tot * par else len(arr) // par
    return pd.DataFrame(arr[:n * par].reshape((n, par)), columns=names)


# ─────────────────────────────────────────────────────────────
# FILENAME PARSER
# EDIT FOR YOUR DATA: this figure takes well metadata from the FCS FILE NAME
# (no plate-map CSV), in the form
#     "<well> <mouse> <condition>[A|B|C]_Experiment.fcs"
# e.g. "E1 M7 381_0.85_Experiment.fcs"   -> strain 381, Rf 0.85, mouse M7
#      "E1 M7 381_0.85B_Experiment.fcs"  -> same, replicate B (optional A/B/C)
#      "E10 M7 dmso_Experiment.fcs"      -> vehicle_ctrl
#      "F1 M7 OVA_Experiment.fcs"        -> antigen_ctrl
#      "G1 M7 EC_store_Experiment.fcs"   -> positive_ctrl
# Change this function if your files are named differently.
# ─────────────────────────────────────────────────────────────

def parse_tcell_filename(fname):
    stem = fname.replace("_Experiment.fcs", "").replace("_Experiment", "")

    if "Well_" in stem:
        return {"well": stem.split(" ")[0], "mouse": "NoDC", "strain": "none",
                "rf_band": None, "replicate": "A",
                "group": "no_dc", "condition": "no_dc"}

    parts = stem.split(" ")
    if len(parts) < 3:
        return None

    well, mouse, cond = parts[0], parts[1], parts[2]
    replicate = "A"
    if cond.endswith(("A", "B", "C")) and "_" in cond:
        replicate, cond = cond[-1], cond[:-1]

    cl = cond.lower()
    if cl in ("dmso", "dmsob"):
        return {"well": well, "mouse": mouse, "strain": "none",
                "rf_band": None, "replicate": replicate,
                "group": "vehicle_ctrl", "condition": "DMSO"}
    if "ova" in cl:
        return {"well": well, "mouse": mouse, "strain": "none",
                "rf_band": None, "replicate": replicate,
                "group": "antigen_ctrl", "condition": "OVA"}
    if "ec_store" in cl or "ec.store" in cl:
        return {"well": well, "mouse": mouse, "strain": "EC_LPXE",
                "rf_band": "store", "replicate": replicate,
                "group": "positive_ctrl", "condition": "EC_store"}

    m = re.match(r"([^_]+)_([0-9.]+)$", cond)
    if m:
        strain_map = {"381": "381", "1587": "1587", "1773": "1773",
                      "ec": "EC_LPXE", "EC": "EC_LPXE"}
        return {"well": well, "mouse": mouse,
                "strain": strain_map.get(m.group(1), m.group(1)),
                "rf_band": float(m.group(2)), "replicate": replicate,
                "group": "lipid_A", "condition": f"{m.group(1)}_{m.group(2)}"}
    return None


# ─────────────────────────────────────────────────────────────
# CTV GATE
# ─────────────────────────────────────────────────────────────
_CTV_GATE = None


def load_ctv_gate(json_path):
    global _CTV_GATE
    with open(json_path) as fh:
        spec = json.load(fh)
    missing = {"channel", "value", "direction"} - set(spec)
    if missing:
        raise ValueError(f"CTV gate JSON missing: {missing}")
    _CTV_GATE = spec
    print(f"CTV gate: {spec['channel']} {spec['direction']} {spec['value']}")


def _apply_gate(df):
    if _CTV_GATE is None or _CTV_GATE["channel"] not in df.columns:
        return df
    val      = float(_CTV_GATE["value"])
    cofactor = float(_CTV_GATE.get("cofactor", 150))
    raw      = df[_CTV_GATE["channel"]].values
    vals     = np.arcsinh(raw / cofactor) if _CTV_GATE.get("transform", "arcsinh") == "arcsinh" else raw
    direction = _CTV_GATE["direction"]
    mask = vals >= val if direction in ("above", "above_threshold", "ge", ">=") else vals <= val
    return df[mask]


# ─────────────────────────────────────────────────────────────
# LOAD FOLDER
# ─────────────────────────────────────────────────────────────

def load_folder(fcs_dir, exp_group):
    fcs_dir   = Path(fcs_dir)
    fcs_files = [f for f in sorted(fcs_dir.glob("*.fcs")) if f.parent == fcs_dir]
    if not fcs_files:
        raise FileNotFoundError(f"No .fcs files in {fcs_dir}")

    print(f"  Loading {len(fcs_files)} files [{exp_group}]...")
    records = []
    for fcs_path in fcs_files:
        meta = parse_tcell_filename(fcs_path.name)
        if meta is None:
            print(f"    ⚠ Could not parse: {fcs_path.name}")
            continue
        try:
            df = read_fcs(str(fcs_path))
        except Exception as e:
            print(f"    ✗ {fcs_path.name}: {e}")
            continue
        if len(df) == 0 or CTV_CHANNEL not in df.columns:
            continue
        for k, v in meta.items():
            df[k] = v
        df["exp_group"] = exp_group
        df = _apply_gate(df)
        if len(df) == 0:
            continue
        records.append(df)

    combined = pd.concat(records, ignore_index=True)
    print(f"    → {len(combined):,} cells, {len(records)} wells")
    return combined


# ─────────────────────────────────────────────────────────────
# SHARED MATH
# ─────────────────────────────────────────────────────────────

def arcsinh(x, cofactor=150):
    return np.arcsinh(np.asarray(x, dtype=float) / cofactor)


def _density(values):
    counts, _ = np.histogram(values, bins=HIST_BINS)
    n = counts.sum()
    return counts / n if n > 0 else counts.astype(float)


def _ddi(sample_vals, vehicle_vals):
    """
    Signed DDI: magnitude = positive area of (sample_density - vehicle_density).
    Sign = +1 if sample median < vehicle median (left shift = proliferation).
    Range [-1, +1].
    """
    if len(sample_vals) < 5 or len(vehicle_vals) < 5:
        return np.nan
    d_s   = _density(sample_vals)
    d_v   = _density(vehicle_vals)
    diff  = d_s - d_v
    mag   = float(np.sum(np.maximum(diff, 0)))
    sign  = -1.0 if np.median(sample_vals) > np.median(vehicle_vals) else 1.0
    return sign * mag


def _pval_label(p):
    if p < 0.001: return "****"
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return "ns"


# ─────────────────────────────────────────────────────────────
# FIGURE 1 — DDI OVERVIEW (two panels: DMSO | Pulse)
# ─────────────────────────────────────────────────────────────

def _compute_ddi_wells(data):
    """
    Compute per-well DDI vs pooled vehicle_ctrl.
    Returns (df_wells, ref_ddi) where ref_ddi maps group_name → ddi value.
    """
    vehicle_df = data[data["group"] == "vehicle_ctrl"]
    if len(vehicle_df) == 0:
        return pd.DataFrame(), {}
    vehicle_vals = arcsinh(vehicle_df[CTV_CHANNEL].dropna().values)

    rows = []
    for (mouse, strain, rf_band, replicate), grp in \
            data[data["group"] == "lipid_A"].groupby(
                ["mouse", "strain", "rf_band", "replicate"]):
        if len(grp) < 5:
            continue
        s_vals = arcsinh(grp[CTV_CHANNEL].dropna().values)
        rows.append({"mouse": mouse, "strain": strain,
                     "rf_band": rf_band, "replicate": replicate,
                     "ddi": _ddi(s_vals, vehicle_vals)})

    ref_ddi = {}
    for grp_name in ["antigen_ctrl", "positive_ctrl"]:
        grp = data[data["group"] == grp_name]
        if len(grp) < 5:
            continue
        g_vals = arcsinh(grp[CTV_CHANNEL].dropna().values)
        ref_ddi[grp_name] = _ddi(g_vals, vehicle_vals)

    return pd.DataFrame(rows) if rows else pd.DataFrame(), ref_ddi


def _draw_ddi_panel(ax, data, group_label, stats_mode="vs_ctrl"):
    df_wells, ref_ddi = _compute_ddi_wells(data)
    if df_wells.empty:
        ax.text(0.5, 0.5, f"No data\n{group_label}",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        return

    strains  = [s for s in STRAIN_ORDER if s in df_wells["strain"].values]
    rf_vals  = sorted(df_wells["rf_band"].dropna().unique())
    rf_to_x  = {rf: i for i, rf in enumerate(rf_vals)}
    n_main   = len(rf_vals)

    n_strains = len(strains)
    strain_offsets = {s: (i - (n_strains - 1) / 2) * 0.18
                      for i, s in enumerate(strains)}

    np.random.seed(42)
    for strain in strains:
        sdf   = df_wells[df_wells["strain"] == strain]
        color = STRAIN_COLORS.get(strain, "#888")
        s_off = strain_offsets.get(strain, 0)

        for _, r in sdf.iterrows():
            if r["rf_band"] is None:
                continue
            val = r["ddi"]
            if np.isnan(val):
                continue
            x      = rf_to_x[r["rf_band"]] + s_off + np.random.uniform(-0.06, 0.06)
            marker = MOUSE_MARKERS.get(r["mouse"], "o")
            ax.scatter(x, val, color=color, marker=marker,
                       s=60, alpha=0.87, zorder=4,
                       edgecolors="white", linewidths=0.5)

        for rf, rfdf in sdf.groupby("rf_band"):
            vals = rfdf["ddi"].dropna().values
            if len(vals) == 0:
                continue
            med = np.median(vals)
            x0  = rf_to_x[rf] + s_off
            ax.hlines(med, x0 - 0.13, x0 + 0.13, color=color, lw=2.8, zorder=5)

    y_data  = df_wells["ddi"].dropna().values
    y_top   = float(max(y_data)) if len(y_data) else 1.0
    y_bot   = float(min(y_data)) if len(y_data) else -1.0
    y_range = (y_top - y_bot) or 1.0

    if stats_mode == "vs_ctrl":
        for strain in strains:
            s_off = strain_offsets.get(strain, 0)
            sdf   = df_wells[df_wells["strain"] == strain]
            for rf, rfdf in sdf.groupby("rf_band"):
                vals = rfdf["ddi"].dropna().values
                if len(vals) < 2 or rf not in rf_to_x:
                    continue
                _, p = stats.ttest_1samp(vals, ref_ddi.get("antigen_ctrl", 0))
                lbl  = _pval_label(p)
                x    = rf_to_x[rf] + s_off
                med  = float(np.median(vals))
                ax.text(x, med + 0.07 * y_range, lbl,
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        color="#c00000" if lbl != "ns" else "#aaaaaa", zorder=7)

    elif stats_mode == "within_strain":
        ref_tops     = [v for v in ref_ddi.values() if not np.isnan(v)]
        effective_top = max([y_top] + ref_tops)
        bracket_top  = effective_top + 0.18 * y_range
        bracket_step = 0.12 * y_range
        max_top      = bracket_top

        for strain in strains:
            s_off = strain_offsets.get(strain, 0)
            sdf   = df_wells[df_wells["strain"] == strain]
            strain_groups = {}
            for rf, rfdf in sdf.groupby("rf_band"):
                vals = rfdf["ddi"].dropna().values
                if len(vals) >= 2:
                    strain_groups[rf] = vals
            if len(strain_groups) < 2:
                continue
            try:
                _, p_anova = stats.f_oneway(*strain_groups.values())
            except Exception:
                continue
            if p_anova >= 0.10:
                continue
            rf_list  = sorted(strain_groups.keys())
            drew_any = False
            for i, rf1 in enumerate(rf_list):
                for rf2 in rf_list[i + 1:]:
                    try:
                        _, p_pair = stats.ttest_ind(
                            strain_groups[rf1], strain_groups[rf2], equal_var=False)
                    except Exception:
                        continue
                    lbl = _pval_label(p_pair)
                    if lbl == "ns":
                        continue
                    if rf1 not in rf_to_x or rf2 not in rf_to_x:
                        continue
                    x1   = rf_to_x[rf1] + s_off
                    x2   = rf_to_x[rf2] + s_off
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

    # antigen_ctrl and pos_ctrl reference lines (vehicle_ctrl = 0 baseline)
    for grp_name, ddi_val in ref_ddi.items():
        if grp_name not in CTRL_STYLES or np.isnan(ddi_val):
            continue
        style = CTRL_STYLES[grp_name]
        ax.axhline(ddi_val, color=style["color"], ls=style["ls"],
                   lw=style["lw"], alpha=0.9, zorder=2)
        ax.text(1.01, ddi_val, style["label"],
                transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=7.5,
                color=style["color"], clip_on=False)

    ax.axhline(0, color="#bbbbbb", lw=0.8, ls="--", alpha=0.5, zorder=1)
    ax.set_xticks(range(n_main))
    ax.set_xticklabels([str(r) for r in rf_vals], fontsize=9)
    ax.set_xlabel("Rf band", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    counts = df_wells.groupby(["strain", "rf_band"])["ddi"].count()
    n_per_band = int(counts.median()) if len(counts) else 0
    n_mice = df_wells["mouse"].nunique()
    ax.text(0.98, 0.02,
            f"N = {n_per_band} per Rf band  ({n_mice} mice)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#555")


def make_figure1(pdf, data):
    lipid_a = data[data["group"] == "lipid_A"]
    strains_present = [s for s in STRAIN_ORDER if s in lipid_a["strain"].values]
    mice_present    = sorted(lipid_a["mouse"].dropna().unique())

    legend_handles = (
        [mlines.Line2D([0], [0], marker=MOUSE_MARKERS.get(m, "o"),
                       color="#555", linestyle="None", markersize=9,
                       label=f"Mouse {m}")
         for m in mice_present]
        + [mlines.Line2D([0], [0], marker="o",
                         color=STRAIN_COLORS.get(s, "#888"),
                         linestyle="None", markersize=9,
                         label=STRAIN_LABELS.get(s, s))
           for s in strains_present]
        + [mlines.Line2D([0], [0], color=CTRL_STYLES[g]["color"],
                         linestyle=CTRL_STYLES[g]["ls"],
                         linewidth=1.8, label=CTRL_STYLES[g]["label"])
           for g in ["antigen_ctrl", "positive_ctrl"]]
        + [mlines.Line2D([0], [0], color="#bbbbbb", linestyle="--",
                         linewidth=1.2, label="0 = vehicle ctrl baseline")]
    )

    for stats_mode, subtitle in [
        ("vs_ctrl",       "one-sample t-test vs antigen_ctrl (OVA)  |  * p<0.10  ** p<0.05  *** p<0.01  **** p<0.001"),
        ("within_strain", "within-strain ANOVA (p<0.10) + pairwise t-test  |  * p<0.10  ** p<0.05  *** p<0.01  **** p<0.001"),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(13, 6.5))
        fig.subplots_adjust(right=0.84)
        _draw_ddi_panel(ax, data, f"DMSO group  ({', '.join(mice_present)})",
                        stats_mode=stats_mode)
        ax.set_ylabel(
            "DDI — signed probability mass\n+ = proliferation   − = suppression",
            fontsize=10)
        fig.suptitle(
            "Figure 1.  OT-II T-Cell Proliferation — Density Difference Integral (DDI)\n"
            f"Vehicle ctrl (DMSO) subtracted as baseline  |  Bar = median  |  {subtitle}",
            fontsize=12, fontweight="bold")
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=min(len(legend_handles), 5),
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
        fig.tight_layout(rect=[0, 0.08, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print("  Figure 1 done (2 pages: vs ctrl + within-strain)")


# ─────────────────────────────────────────────────────────────
# FIGURE 2 — RESIDUAL HISTOGRAM GRIDS (one page per mouse)
# ─────────────────────────────────────────────────────────────

def _draw_hist_panel(ax, strain, rf_band, mouse, lipid_a,
                     vehicle_vals, ova_residual, ec_residual):
    # OVA reference — grey, every panel
    if ova_residual is not None:
        ax.step(HIST_CENTERS, ova_residual, where="mid",
                color=CTRL_STYLES["antigen_ctrl"]["color"],
                lw=CTRL_STYLES["antigen_ctrl"]["lw"],
                ls=CTRL_STYLES["antigen_ctrl"]["ls"],
                alpha=0.85, label="OVA (antigen ctrl)", zorder=2)

    # pos_ctrl reference — orange dotted, every panel
    if ec_residual is not None:
        ax.step(HIST_CENTERS, ec_residual, where="mid",
                color=CTRL_STYLES["positive_ctrl"]["color"],
                lw=CTRL_STYLES["positive_ctrl"]["lw"],
                ls=CTRL_STYLES["positive_ctrl"]["ls"],
                alpha=0.75, label="pos_ctrl", zorder=2)

    if strain == "pos_ctrl":
        # Controls reference panel — fill both for visibility
        if ova_residual is not None:
            ax.fill_between(HIST_CENTERS, ova_residual,
                            where=ova_residual > 0, alpha=0.20,
                            color=CTRL_STYLES["antigen_ctrl"]["color"], step="mid")
            ax.fill_between(HIST_CENTERS, ova_residual,
                            where=ova_residual < 0, alpha=0.20,
                            color=CTRL_STYLES["antigen_ctrl"]["color"], step="mid")
        if ec_residual is not None:
            ax.fill_between(HIST_CENTERS, ec_residual,
                            where=ec_residual > 0, alpha=0.18,
                            color=CTRL_STYLES["positive_ctrl"]["color"], step="mid")
            ax.fill_between(HIST_CENTERS, ec_residual,
                            where=ec_residual < 0, alpha=0.18,
                            color=CTRL_STYLES["positive_ctrl"]["color"], step="mid")
        ax.set_title("Controls reference", fontsize=8.5,
                     fontweight="bold", color="#444")

    else:
        color = STRAIN_COLORS.get(strain, "#888")
        try:
            rf_str = f"Rf {float(rf_band):.3f}"
        except (ValueError, TypeError):
            rf_str = f"Rf {rf_band}"

        mouse_df = lipid_a[
            (lipid_a["strain"] == strain) &
            (lipid_a["rf_band"] == rf_band) &
            (lipid_a["mouse"] == mouse)
        ]

        for replicate, rep_grp in mouse_df.groupby("replicate"):
            if len(rep_grp) < 5:
                continue
            resid = _density(arcsinh(rep_grp[CTV_CHANNEL].dropna().values)) - \
                    _density(vehicle_vals)
            style = REP_STYLES.get(replicate, {"lw": 1.6, "alpha": 0.8, "ls": "-"})
            ax.fill_between(HIST_CENTERS, resid,
                            where=resid > 0, alpha=0.15,
                            color=color, step="mid", zorder=3)
            ax.fill_between(HIST_CENTERS, resid,
                            where=resid < 0, alpha=0.15,
                            color=color, step="mid", zorder=3)
            ax.step(HIST_CENTERS, resid, where="mid",
                    color=color, lw=style["lw"],
                    alpha=style["alpha"], ls=style["ls"],
                    label=f"Rep {replicate}", zorder=4)

        ax.set_title(f"{STRAIN_LABELS.get(strain, strain)}  |  {rf_str}",
                     fontsize=8.5, fontweight="bold", color=color)

    ax.axhline(0, color="#333", lw=0.9, ls="--", alpha=0.55, zorder=1)
    ax.set_xlabel("CTV (arcsinh)", fontsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, framealpha=0.6, loc="upper right", handlelength=1.5)


def _residual_grid_for_mouse(pdf, mouse, group_label, data):
    vehicle_df = data[data["group"] == "vehicle_ctrl"]
    if len(vehicle_df) == 0 or CTV_CHANNEL not in vehicle_df.columns:
        print(f"  ⚠ No vehicle_ctrl for {group_label} mouse {mouse}, skipping")
        return

    vehicle_vals = arcsinh(vehicle_df[CTV_CHANNEL].dropna().values)
    lipid_a      = data[data["group"] == "lipid_A"]

    ova_df = data[data["group"] == "antigen_ctrl"]
    ec_df  = data[data["group"] == "positive_ctrl"]

    ova_residual = None
    if len(ova_df) >= 5 and CTV_CHANNEL in ova_df.columns:
        ova_residual = _density(arcsinh(ova_df[CTV_CHANNEL].dropna().values)) - \
                       _density(vehicle_vals)

    ec_residual = None
    if len(ec_df) >= 5 and CTV_CHANNEL in ec_df.columns:
        ec_residual = _density(arcsinh(ec_df[CTV_CHANNEL].dropna().values)) - \
                      _density(vehicle_vals)

    def rf_panels(strain):
        sdf = lipid_a[lipid_a["strain"] == strain]
        return [(strain, rf) for rf in sorted(sdf["rf_band"].dropna().unique())]

    row_a = [("pos_ctrl", None)] + rf_panels("381")
    row_b = rf_panels("1587") + rf_panels("1773")

    while len(row_a) < 4:
        row_a.append(None)
    while len(row_b) < 4:
        row_b.append(None)

    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharey=True, squeeze=False)
    fig.suptitle(
        f"Figure 2.  Residual Histograms — {group_label} — Mouse {mouse}\n"
        "Grey = OVA (antigen ctrl) ref  |  "
        "Orange dotted = pos_ctrl ref  |  "
        "Y = 0 = vehicle ctrl baseline (DMSO subtracted)",
        fontsize=11, fontweight="bold")

    for col, panel in enumerate(row_a):
        ax = axes[0][col]
        if panel is None:
            ax.axis("off")
        else:
            _draw_hist_panel(ax, panel[0], panel[1], mouse,
                             lipid_a, vehicle_vals, ova_residual, ec_residual)

    for col, panel in enumerate(row_b):
        ax = axes[1][col]
        if panel is None:
            ax.axis("off")
        else:
            _draw_hist_panel(ax, panel[0], panel[1], mouse,
                             lipid_a, vehicle_vals, ova_residual, ec_residual)

    axes[0][0].set_ylabel("Density diff\n(sample − vehicle ctrl)", fontsize=8)
    axes[1][0].set_ylabel("Density diff\n(sample − vehicle ctrl)", fontsize=8)

    axes[0][0].annotate("Row A\n(WT + pos ctrl)",
                        xy=(-0.22, 0.5), xycoords="axes fraction",
                        ha="center", va="center", fontsize=8,
                        color="#555", rotation=90)
    axes[1][0].annotate("Row B\n(phos muts)",
                        xy=(-0.22, 0.5), xycoords="axes fraction",
                        ha="center", va="center", fontsize=8,
                        color="#555", rotation=90)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig, bbox_inches="tight", dpi=120)
    plt.close(fig)


def make_figure2(pdf, data):
    lipid_a = data[data["group"] == "lipid_A"]
    mice    = sorted(lipid_a["mouse"].unique()) if len(lipid_a) > 0 else []
    print(f"  Mice: {', '.join(mice)}")
    for mouse in mice:
        _residual_grid_for_mouse(pdf, mouse, "DMSO group", data)
    print("  Figure 2 done")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TAC figures for OT-II T-cell proliferation data")
    parser.add_argument("--data",     required=True,
                        help="Folder: gated FCS files (e.g. DMSO_mid)")
    parser.add_argument("--out",      required=True,
                        help="Output PDF path")
    parser.add_argument("--ctv-gate", default=None, dest="ctv_gate",
                        help="Optional CTV threshold gate JSON")
    args = parser.parse_args()

    if args.ctv_gate:
        load_ctv_gate(args.ctv_gate)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nLoading FCS files...")
    data = load_folder(args.data, "DMSO group")

    print(f"\nWriting PDF → {out_path}")
    with PdfPages(str(out_path)) as pdf:
        print("\nFigure 1 — DDI overview...")
        make_figure1(pdf, data)

        print("\nFigure 2 — Residual histogram grids...")
        make_figure2(pdf, data)

    print(f"\nDone. PDF saved → {out_path}")


if __name__ == "__main__":
    main()
