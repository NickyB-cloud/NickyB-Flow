#!/usr/bin/env python3
"""
make_gate_qc_pdf_GitHub.py
Generates a QC PDF showing the full gating sequence for every well.
One page per well, four panels showing each gate step.

Run after all gates have been applied with Apply_sequential_gates_GitHub.py.

Usage:
    python make_gate_qc_pdf_GitHub.py \
        --pre-gate        results/unmixed/Exp1 \
        --post-live       analysis/Exp1/gated/Exp1_live \
        --post-cd11c      analysis/Exp1/gated/Exp1_cd11cpos \
        --post-bmdcs      analysis/Exp1/gated/Exp1_BMDCs \
        --post-bmdms      analysis/Exp1/gated/Exp1_BMDMs \
        --ld-threshold    analysis/Exp1/gates/LD_gate_threshold.json \
        --cd11c-threshold analysis/Exp1/gates/CD11cpos_gate_threshold.json \
        --bmdcs-hull      analysis/Exp1/gates/BMDCs_gate_hull.txt \
        --bmdms-hull      analysis/Exp1/gates/BMDMs_gate_hull.txt \
        --cd11c-channel   BUV395 \
        --mhcii-channel   PERCP \
        --out             results/plots/Exp1_gating_QC.pdf

Panel layout per well (one page):
    [FSC vs SSC] [Ghost Dye viability] [CD11c+] [MHC-II vs CD11b]
    grey = all events entering that gate
    color = events that passed
    gate boundary drawn on top
    % retained annotated
"""

import argparse
import json
import os
import struct
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# FCS READER — standalone pure-Python implementation
# ─────────────────────────────────────────────────────────────

def parse_text_segment(text_block):
    if not text_block:
        return {}
    delim = text_block[0]
    placeholder = "\x00"
    text_block = text_block.replace(delim * 2, placeholder)
    parts = text_block.strip(delim).split(delim)
    parts = [p.replace(placeholder, delim) for p in parts]
    text_dict = {}
    for i in range(0, len(parts) - 1, 2):
        key = parts[i].strip().lower()
        val = parts[i + 1].strip()
        text_dict[key] = val
    return text_dict


def read_fcs(path):
    with open(path, "rb") as f:
        raw = f.read()
    header     = raw[:58].decode("ascii", errors="ignore")
    text_start = int(header[10:18].strip() or 0)
    text_end   = int(header[18:26].strip() or 0)
    data_start = int(header[26:34].strip() or 0)
    data_end   = int(header[34:42].strip() or 0)
    text_block = raw[text_start:text_end + 1].decode("ascii", errors="ignore")
    t = parse_text_segment(text_block)

    def meta(key, default=None):
        return t.get(key, t.get(f"${key}", default))

    data_start = int(meta("begindata", data_start))
    data_end   = int(meta("enddata",   data_end))
    par = int(meta("par", 0))
    tot = int(meta("tot", 0))
    data = raw[data_start:data_end + 1]

    dtype   = str(meta("datatype", "F")).upper()
    byteord = str(meta("byteord", "1,2,3,4"))
    little  = byteord.startswith("1")

    if dtype == "F":
        dt = np.dtype("<f4") if little else np.dtype(">f4")
    elif dtype == "D":
        dt = np.dtype("<f8") if little else np.dtype(">f8")
    else:
        bits = int(meta("p1b", 32))
        endian = "<" if little else ">"
        dt = np.dtype(f"{endian}u{bits // 8}")

    arr = np.frombuffer(data, dtype=dt)
    channel_names = []
    for i in range(1, par + 1):
        candidates = [meta(f"p{i}n"), meta(f"p{i}s")]
        name = next((c for c in candidates if c and c.strip()), f"Ch{i}")
        channel_names.append(name)

    if tot > 0 and len(arr) == tot * par:
        events = arr.reshape((tot, par))
    else:
        n = len(arr) // par
        events = arr[:n * par].reshape((n, par))

    return pd.DataFrame(events, columns=channel_names)


# ─────────────────────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────────────────────

def arcsinh(x, cofactor=150):
    return np.arcsinh(np.asarray(x, dtype=float) / cofactor)


# ─────────────────────────────────────────────────────────────
# GATE LOADERS
# ─────────────────────────────────────────────────────────────

def load_threshold(json_path):
    with open(json_path) as f:
        return json.load(f)


def load_hull(txt_path):
    coords = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if line:
                coords.append(eval(line))
    if len(coords) > 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return coords


# ─────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────

def scatter_gate_panel(ax, df_all, df_passed, x_col, y_col,
                       title, pct, transform=True,
                       gate_hull=None, threshold=None,
                       threshold_direction=None):
    """
    Plot all events in grey, passed events in color.
    Optionally draw gate boundary.
    """
    # Transform
    if transform:
        if x_col in df_all.columns:
            x_all = arcsinh(df_all[x_col].values)
        else:
            ax.text(0.5, 0.5, f"{x_col}\nnot found",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(title, fontsize=9)
            return
        if y_col in df_all.columns:
            y_all = arcsinh(df_all[y_col].values)
        else:
            ax.text(0.5, 0.5, f"{y_col}\nnot found",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(title, fontsize=9)
            return
        x_pass = arcsinh(df_passed[x_col].values) if x_col in df_passed.columns else np.array([])
        y_pass = arcsinh(df_passed[y_col].values) if y_col in df_passed.columns else np.array([])
        xlabel = f"{x_col} (arcsinh)"
        ylabel = f"{y_col} (arcsinh)"
    else:
        x_all  = df_all[x_col].values  if x_col in df_all.columns  else np.array([])
        y_all  = df_all[y_col].values  if y_col in df_all.columns  else np.array([])
        x_pass = df_passed[x_col].values if x_col in df_passed.columns else np.array([])
        y_pass = df_passed[y_col].values if y_col in df_passed.columns else np.array([])
        xlabel = x_col
        ylabel = y_col

    # Subsample for speed
    max_pts = 3000
    def subsample(x, y, n):
        if len(x) > n:
            idx = np.random.choice(len(x), n, replace=False)
            return x[idx], y[idx]
        return x, y

    x_all_s,  y_all_s  = subsample(x_all,  y_all,  max_pts)
    x_pass_s, y_pass_s = subsample(x_pass, y_pass, max_pts)

    # Plot grey background (all events)
    ax.scatter(x_all_s, y_all_s,
               s=1, color="#cccccc", alpha=0.5, rasterized=True, zorder=1)

    # Plot colored passed events
    ax.scatter(x_pass_s, y_pass_s,
               s=1, color="#2166ac", alpha=0.6, rasterized=True, zorder=2)

    # Draw polygon gate
    if gate_hull and len(gate_hull) >= 3:
        poly_patch = MplPolygon(
            gate_hull, closed=True,
            fill=False, edgecolor="#e6550d", linewidth=1.5, linestyle="--", zorder=3
        )
        ax.add_patch(poly_patch)

    # Draw threshold line
    if threshold is not None:
        ax.axvline(threshold, color="#e6550d", lw=1.5, ls="--", zorder=3)

    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_title(f"{title}\n{len(df_passed):,}/{len(df_all):,} ({pct:.1f}%)",
                 fontsize=8, pad=3)
    ax.tick_params(labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def histogram_gate_panel(ax, df_all, df_passed, col,
                         title, pct, threshold=None, direction=None):
    """
    Histogram with all events grey, passed events colored.
    Threshold line drawn if provided.
    """
    if col not in df_all.columns:
        ax.text(0.5, 0.5, f"{col}\nnot found",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title(title, fontsize=9)
        return

    all_vals  = arcsinh(df_all[col].values)
    pass_vals = arcsinh(df_passed[col].values) if col in df_passed.columns else np.array([])

    all_vals  = all_vals[np.isfinite(all_vals)]
    pass_vals = pass_vals[np.isfinite(pass_vals)]

    bins = np.linspace(
        np.nanmin(all_vals) if len(all_vals) else -1,
        np.nanmax(all_vals) if len(all_vals) else 8,
        120
    )

    ax.hist(all_vals,  bins=bins, color="#cccccc", alpha=0.8,
            density=True, label="All", zorder=1)
    ax.hist(pass_vals, bins=bins, color="#2166ac", alpha=0.7,
            density=True, label="Passed", zorder=2)

    if threshold is not None:
        ax.axvline(threshold, color="#e6550d", lw=1.5, ls="--",
                   label=f"threshold={threshold:.2f}", zorder=3)

    ax.set_xlabel(f"{col} (arcsinh)", fontsize=7)
    ax.set_ylabel("Density", fontsize=7)
    ax.set_title(f"{title}\n{len(pass_vals):,}/{len(all_vals):,} ({pct:.1f}%)",
                 fontsize=8, pad=3)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, framealpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def pct(n_after, n_before):
    return 100 * n_after / n_before if n_before > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-well gating QC PDF."
    )
    parser.add_argument("--pre-gate",   required=True,
                        help="Folder before any gating (unmixed FCS)")
    parser.add_argument("--post-live",  required=True,
                        help="Folder after viability gate")
    parser.add_argument("--post-cd11c", required=True,
                        help="Folder after CD11c gate")
    parser.add_argument("--post-bmdcs", required=True,
                        help="Folder after BMDCs polygon gate")
    parser.add_argument("--post-bmdms", required=True,
                        help="Folder after BMDMs polygon gate")
    parser.add_argument("--gates",         required=False, default=None,
                        help="Folder containing gate files (JSON + hull TXT). "
                             "Used as base path when individual gate args are not provided.")
    parser.add_argument("--ld-threshold",  required=False, default=None,
                        help="Path to live/dead threshold JSON. "
                             "Overrides --gates/live.dead_gate_threshold.json")
    parser.add_argument("--cd11c-threshold", required=False, default=None,
                        help="Path to CD11c threshold JSON. "
                             "Overrides --gates/CD11C+_gate_threshold.json")
    parser.add_argument("--bmdcs-hull",    required=False, default=None,
                        help="Path to BMDCs hull TXT. "
                             "Overrides --gates/BMDCs_gate_hull.txt")
    parser.add_argument("--bmdms-hull",    required=False, default=None,
                        help="Path to BMDMs hull TXT. "
                             "Overrides --gates/BMDMs_gate_hull.txt")
    parser.add_argument("--out",           required=True,
                        help="Output PDF path")
    parser.add_argument("--cd11c-channel", default="BUV395",
                        help="FCS channel name for CD11c (default: BUV395; EDIT to match your panel)")
    parser.add_argument("--mhcii-channel", default="PERCP",
                        help="FCS channel name for MHC-II (default: PERCP; EDIT to match your panel)")
    parser.add_argument("--title",         default="Gating QC Report",
                        help="Title shown on cover page")
    args = parser.parse_args()

    # ── Resolve gate file paths ──────────────────────────────
    # Explicit args take priority; fall back to --gates folder with default names
    gates_dir = Path(args.gates) if args.gates else None

    def resolve_gate(explicit, default_name):
        if explicit:
            return Path(explicit)
        if gates_dir and default_name:
            return gates_dir / default_name
        raise ValueError(
            f"Must provide either --gates folder or explicit path for {default_name}")

    ld_path    = resolve_gate(args.ld_threshold,    "live.dead_gate_threshold.json")
    cd11c_path = resolve_gate(args.cd11c_threshold, "CD11C+_gate_threshold.json")
    bmdcs_path = resolve_gate(args.bmdcs_hull,      "BMDCs_gate_hull.txt")
    bmdms_path = resolve_gate(args.bmdms_hull,      "BMDMs_gate_hull.txt")

    viability_spec = load_threshold(str(ld_path))
    cd11c_spec     = load_threshold(str(cd11c_path))
    bmdcs_hull     = load_hull(str(bmdcs_path))
    bmdms_hull     = load_hull(str(bmdms_path))

    viability_threshold = viability_spec["value"]
    cd11c_threshold     = cd11c_spec["value"]

    # ── Map folders ──────────────────────────────────────────
    folders = {
        "pre":    Path(args.pre_gate),
        "live":   Path(args.post_live),
        "cd11c":  Path(args.post_cd11c),
        "bmdcs":  Path(args.post_bmdcs),
        "bmdms":  Path(args.post_bmdms),
    }

    # ── Find all wells from pre-gate folder ──────────────────
    # Match by well prefix (e.g. "A1", "E6")
    pre_files = sorted(folders["pre"].glob("*.fcs"))

    # Exclude DMSO/ and OVA/ subfolders — only direct .fcs files
    pre_files = [f for f in pre_files if f.parent == folders["pre"]]

    if not pre_files:
        print(f"❌ No FCS files found in {folders['pre']}")
        return

    print(f"\nGenerating gating QC PDF for {len(pre_files)} wells...")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    with PdfPages(str(out_path)) as pdf:

        # ── Cover page ───────────────────────────────────────
        fig, ax = plt.subplots(figsize=(13, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.75, args.title,
                ha="center", va="center", fontsize=20, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.62, "One page per well | Four panels per page",
                ha="center", va="center", fontsize=13, color="#555",
                transform=ax.transAxes)

        gate_lines = [
            ("Gate 1", "FSC vs SSC", "cell_125x_hull.txt", "#444444"),
            ("Gate 2", f"Viability — Ghost Dye ≤ {viability_threshold:.3f} (arcsinh)", "", "#2166ac"),
            ("Gate 3", f"CD11c+ — {args.cd11c_channel} ≥ {cd11c_threshold:.3f} (arcsinh)", "", "#4dac26"),
            ("Gate 4a", "BMDCs — MHC-II hi CD11b int (GM-DC polygon)", "", "#e6550d"),
            ("Gate 4b", "BMDMs — MHC-II int CD11b hi (GM-Mac polygon)", "", "#8B008B"),
        ]

        for i, (gate, desc, note, color) in enumerate(gate_lines):
            y = 0.48 - i * 0.07
            ax.text(0.15, y, gate, ha="left", fontsize=11,
                    fontweight="bold", color=color, transform=ax.transAxes)
            ax.text(0.28, y, desc, ha="left", fontsize=11,
                    color="#333", transform=ax.transAxes)

        ax.text(0.5, 0.10,
                "Grey = all events entering gate  |  Blue = events that passed  |  "
                "Orange dashed = gate boundary",
                ha="center", fontsize=10, color="#666", transform=ax.transAxes)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── One page per well ────────────────────────────────
        for fcs_path in pre_files:
            well = fcs_path.name.split(" ")[0]  # e.g. "A1"

            # Find matching file in each downstream folder
            def find_well(folder, well):
                matches = list(folder.glob(f"{well} *.fcs"))
                return matches[0] if matches else None

            f_live  = find_well(folders["live"],  well)
            f_cd11c = find_well(folders["cd11c"], well)
            f_bmdcs = find_well(folders["bmdcs"], well)
            f_bmdms = find_well(folders["bmdms"], well)

            # Load dataframes — use empty df if file missing (0 events after gate)
            try:
                df_pre = read_fcs(str(fcs_path))
            except Exception as e:
                print(f"  ✗ {well} — could not read pre-gate file: {e}")
                continue

            df_live  = read_fcs(str(f_live))  if f_live  else pd.DataFrame(columns=df_pre.columns)
            df_cd11c = read_fcs(str(f_cd11c)) if f_cd11c else pd.DataFrame(columns=df_pre.columns)
            df_bmdcs = read_fcs(str(f_bmdcs)) if f_bmdcs else pd.DataFrame(columns=df_pre.columns)
            df_bmdms = read_fcs(str(f_bmdms)) if f_bmdms else pd.DataFrame(columns=df_pre.columns)

            n_pre   = len(df_pre)
            n_live  = len(df_live)
            n_cd11c = len(df_cd11c)
            n_bmdcs = len(df_bmdcs)
            n_bmdms = len(df_bmdms)

            # ── Build page: 1 row, 5 panels ──────────────────
            # Panel 1: FSC vs SSC (pre → live)
            # Panel 2: Ghost Dye histogram (pre → live)
            # Panel 3: CD11c histogram (live → cd11c)
            # Panel 4: MHC-II vs CD11b — BMDCs (cd11c → bmdcs)
            # Panel 5: MHC-II vs CD11b — BMDMs (cd11c → bmdms)

            fig, axes = plt.subplots(1, 5, figsize=(18, 4))
            fig.suptitle(
                f"Well {well}  |  "
                f"Pre-gate: {n_pre:,}  →  "
                f"Live: {n_live:,} ({pct(n_live, n_pre):.0f}%)  →  "
                f"CD11c+: {n_cd11c:,} ({pct(n_cd11c, n_pre):.0f}% of pre)  →  "
                f"BMDCs: {n_bmdcs:,}  BMDMs: {n_bmdms:,}",
                fontsize=9, fontweight="bold"
            )

            # Panel 1 — FSC vs SSC
            fsc_col = next((c for c in df_pre.columns if "FSC-A" in c or c == "FSC-A"), None)
            ssc_col = next((c for c in df_pre.columns if "SSC-A" in c or c == "SSC-A"), None)
            if fsc_col and ssc_col:
                scatter_gate_panel(
                    axes[0], df_pre, df_live,
                    fsc_col, ssc_col,
                    "Gate 1: FSC vs SSC",
                    pct(n_live, n_pre),
                    transform=False,
                )
            else:
                axes[0].text(0.5, 0.5, "FSC/SSC\nnot found",
                             ha="center", va="center",
                             transform=axes[0].transAxes, fontsize=9)
                axes[0].set_title("Gate 1: FSC vs SSC", fontsize=8)

            # Panel 2 — Viability histogram
            histogram_gate_panel(
                axes[1], df_pre, df_live,
                "GHOST DYE VIOLET 540",
                "Gate 2: Viability",
                pct(n_live, n_pre),
                threshold=viability_threshold,
                direction="below",
            )

            # Panel 3 — CD11c histogram
            histogram_gate_panel(
                axes[2], df_live, df_cd11c,
                args.cd11c_channel,
                "Gate 3: CD11c+",
                pct(n_cd11c, n_live),
                threshold=cd11c_threshold,
                direction="above",
            )

            # Panel 4 — MHC-II vs CD11b: BMDCs
            scatter_gate_panel(
                axes[3], df_cd11c, df_bmdcs,
                args.mhcii_channel, "APC",
                "Gate 4a: BMDCs",
                pct(n_bmdcs, n_cd11c),
                transform=True,
                gate_hull=bmdcs_hull,
            )

            # Panel 5 — MHC-II vs CD11b: BMDMs
            scatter_gate_panel(
                axes[4], df_cd11c, df_bmdms,
                args.mhcii_channel, "APC",
                "Gate 4b: BMDMs",
                pct(n_bmdms, n_cd11c),
                transform=True,
                gate_hull=bmdms_hull,
            )
            # Color BMDMs purple to distinguish from BMDCs
            axes[4].collections[1].set_color("#8B008B") if len(axes[4].collections) > 1 else None

            plt.tight_layout(rect=[0, 0, 1, 0.92])
            pdf.savefig(fig, bbox_inches="tight", dpi=120)
            plt.close(fig)

            print(f"  ✓ {well}  |  "
                  f"{n_pre:,} → live {n_live:,} → "
                  f"CD11c+ {n_cd11c:,} → "
                  f"BMDCs {n_bmdcs:,} / BMDMs {n_bmdms:,}")

    print(f"\nDone. PDF saved → {out_path}")


if __name__ == "__main__":
    main()