#unmixer_GitHub.py — Unified Δ‑math + spectral unmixing builder
#!/usr/bin/env python3
"""
unmixer_GitHub.py — Unified Δ‑math + spectral unmixing builder

Consumes:
    ExperimentStructure (from organizer_GitHub.py)

Produces:
    delta_table.txt
    mixing_matrix.tsv
    unmixing_matrix.tsv
    unmixing_QC.pdf

Handles:
    - bead SS/FMO pairs
    - cell SS/FMO pairs
    - cell SS with missing FMOs (e.g., Ghost Violet)
    - pooled unstained controls
    - missing unstained gracefully

Patches applied:
    1. load_events is cached via lru_cache — unstained is no longer re-read per fluor
    2. compute_delta validates channel alignment across SS/FMO/unstained
    3. build_mixing_matrix warns on all-zero spectral vectors
    4. channel_order is sorted for deterministic matrix output
    5. delta_dict stores (source, rows) tuples — replaces fragile `is` identity check
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from functools import lru_cache
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

from organizer_GitHub import organize, ExperimentStructure, ControlFile, ControlPair
from fcs_reader_GitHub import read_fcs

# Channels we never want to treat as fluorescence detectors
EXCLUDED_CHANNELS = {
    "FSC-A", "FSC-H",
    "SSC-A", "SSC-H", "SSC-B-A", "SSC-B-H",
    "TIME",
}


# ============================================================
# 1. Δ‑MATH HELPERS
# ============================================================

# PATCH 1: Cache load_events so the unstained file is not re-read once per fluor.
# lru_cache keys on the Path object — requires Path to be hashable (it is).
@lru_cache(maxsize=None)
def load_events_cached(path: Path):
    events, labels, _ = read_fcs(path)
    return events, tuple(labels)  # tuple so the return value is hashable


def load_events(path: Path):
    events, labels = load_events_cached(path)

    # Filter out excluded channels
    keep_indices = [i for i, label in enumerate(labels) if label.upper() not in EXCLUDED_CHANNELS]

    filtered_events = events[:, keep_indices]
    filtered_labels = [labels[i] for i in keep_indices]

    return filtered_events, filtered_labels



def compute_delta(
    ss: ControlFile,
    fmo: ControlFile | None,
    unstained: ControlFile | None,
) -> list[dict]:
    """
    Compute Δ1 and Δ2 for one fluor using MEDIAN-BASED Δ-math:

        Δ1 = median(SS)  - median(UN)
        Δ2 = median(FMO) - median(UN)

    This avoids event-wise subtraction (which fails when event counts differ)
    and matches Cytek's recommended spectral unmixing math.

    Returns a list of per-channel dicts with medians, p99s, and channel names.
    """

    ss_events, labels = load_events(ss.path)

    # PATCH 2: Validate channel alignment before doing any arithmetic.
    if unstained is not None:
        un_labels = load_events(unstained.path)[1]
        if un_labels != labels:
            raise ValueError(
                f"Channel mismatch between SS ({ss.path.name}) "
                f"and unstained ({unstained.path.name}):\n"
                f"  SS:        {labels}\n"
                f"  Unstained: {un_labels}"
            )

    if fmo is not None:
        fmo_labels = load_events(fmo.path)[1]
        if fmo_labels != labels:
            raise ValueError(
                f"Channel mismatch between SS ({ss.path.name}) "
                f"and FMO ({fmo.path.name}):\n"
                f"  SS:  {labels}\n"
                f"  FMO: {fmo_labels}"
            )

    # Optional controls
    un_events = load_events(unstained.path)[0] if unstained else None
    fmo_events = load_events(fmo.path)[0] if fmo else None

    rows = []
    for i, ch in enumerate(labels):
        ss_vals = ss_events[:, i]

        # Medians
        ss_med = float(np.median(ss_vals))
        ss_p99 = float(np.percentile(ss_vals, 99))

        # Unstained
        if un_events is not None:
            un_vals = un_events[:, i]
            un_med = float(np.median(un_vals))
            un_p99 = float(np.percentile(un_vals, 99))
        else:
            un_vals = None
            un_med = None
            un_p99 = None

        # FMO
        if fmo_events is not None:
            fmo_vals = fmo_events[:, i]
            fmo_med = float(np.median(fmo_vals))
            fmo_p99 = float(np.percentile(fmo_vals, 99))
        else:
            fmo_vals = None
            fmo_med = None
            fmo_p99 = None

        # Δ‑math (median-based)
        d1_med = ss_med - (un_med if un_med is not None else 0)
        d2_med = fmo_med - (un_med if un_med is not None else 0) if fmo_med is not None else None

        # p99 deltas (same logic)
        d1_p99 = ss_p99 - (un_p99 if un_p99 is not None else 0)
        d2_p99 = fmo_p99 - (un_p99 if un_p99 is not None else 0) if fmo_p99 is not None else None

        rows.append({
            "channel": ch,
            "ss_med": ss_med,
            "un_med": un_med,
            "fmo_med": fmo_med,
            "d1_med": d1_med,
            "d2_med": d2_med,
            "ss_p99": ss_p99,
            "un_p99": un_p99,
            "fmo_p99": fmo_p99,
            "d1_p99": d1_p99,
            "d2_p99": d2_p99,
        })

    return rows


# ============================================================
# 2. DELTA TABLE WRITER
# ============================================================

def _fmt(val) -> str:
    """Format a numeric-or-None value for TSV output."""
    return "NA" if val is None else str(val)


def write_delta_table(delta_dict: dict[str, tuple[str, list[dict]]], outpath: Path) -> None:
    """Write per-fluor Δ statistics to a tab-separated text file."""
    lines = []
    for fluor, (source, rows) in delta_dict.items():
        lines.append(f"=== {fluor} (source: {source}) ===")
        lines.append(
            "Channel\tSS_med\tUN_med\tFMO_med\tΔ1_med\tΔ2_med"
            "\tSS_p99\tUN_p99\tFMO_p99\tΔ1_p99\tΔ2_p99"
        )
        for r in rows:
            lines.append(
                f"{r['channel']}\t"
                f"{_fmt(r['ss_med'])}\t{_fmt(r['un_med'])}\t{_fmt(r['fmo_med'])}\t"
                f"{_fmt(r['d1_med'])}\t{_fmt(r['d2_med'])}\t"
                f"{_fmt(r['ss_p99'])}\t{_fmt(r['un_p99'])}\t{_fmt(r['fmo_p99'])}\t"
                f"{_fmt(r['d1_p99'])}\t{_fmt(r['d2_p99'])}"
            )
        lines.append("")

    outpath.write_text("\n".join(lines))


# ============================================================
# 3. SPECTRAL VECTOR EXTRACTION
# ============================================================

def extract_spectral_vector(
    delta_rows: list[dict],
    channel_order: list[str],
) -> np.ndarray:
    """
    Build a Δ1_p99 vector aligned to channel_order.
    Channels absent in delta_rows are filled with 0.
    """
    vec = np.zeros(len(channel_order))
    ch_index = {ch: i for i, ch in enumerate(channel_order)}
    for r in delta_rows:
        idx = ch_index.get(r["channel"])
        if idx is not None:
            vec[idx] = max(0.0, r["d1_p99"])
    return vec


# ============================================================
# 4. MATRIX BUILDERS
# ============================================================

def build_mixing_matrix(
    spectral_vectors: dict[str, np.ndarray],
    fluor_order: list[str],
) -> np.ndarray:
    rows = []
    for fluor in fluor_order:
        v = spectral_vectors[fluor].copy()
        if v.max() > 0:
            v = v / v.max()
        else:
            # PATCH 3: Warn instead of silently producing a zero row, which
            # causes np.linalg.pinv to produce garbage for this fluor's column.
            warnings.warn(
                f"Fluor {fluor!r} has an all-zero spectral vector — "
                "it will produce a zero row in the mixing matrix and "
                "unreliable unmixing results. Check your SS controls.",
                UserWarning,
            )
        rows.append(v)
    return np.stack(rows, axis=0)


def build_unmixing_matrix(M: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(M.T)


# ============================================================
# 5. MAIN UNMIXING PIPELINE
# ============================================================

def run_unmixing(structure: ExperimentStructure, outdir: Path) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Build separate bead and cell delta tables ---
    bead_delta: dict[str, list[dict]] = {}
    cell_delta: dict[str, list[dict]] = {}

    # Bead SS+FMO
    for pair in structure.beads.pairs:
        bead_delta[pair.fluor] = compute_delta(
            ss=pair.ss,
            fmo=pair.fmo,
            unstained=structure.beads.unstained,
        )

    # Cell SS+FMO
    for pair in structure.cells.pairs:
        cell_delta[pair.fluor] = compute_delta(
            ss=pair.ss,
            fmo=pair.fmo,
            unstained=structure.cells.unstained,
        )

    # Cell SS-only (no FMO)
    for ss in structure.cells.missing_fmo:
        cell_delta[ss.fluor] = compute_delta(
            ss=ss,
            fmo=None,
            unstained=structure.cells.unstained,
        )

    # --- Apply override rule: p99‑based, top‑2 channels only ---
    # PATCH: Instead of scanning all channels or using a primary detector,
    # we rank channels by SS_p99 and evaluate Δ2_med > Δ1_med only on the top 2.

    delta_dict: dict[str, tuple[str, list[dict]]] = {}

    for fluor in set(bead_delta.keys()) | set(cell_delta.keys()):
        # If only cell exists → use cell
        if fluor not in bead_delta:
            delta_dict[fluor] = ("cell", cell_delta[fluor])
            continue

        # If only bead exists → use bead
        if fluor not in cell_delta:
            delta_dict[fluor] = ("bead", bead_delta[fluor])
            continue

        # Both exist → apply top‑2 p99 override rule
        rows = cell_delta[fluor]

        # Sort channels by SS_p99 descending
        sorted_rows = sorted(
            rows,
            key=lambda r: (r.get("ss_p99") or 0),
            reverse=True
        )

        # Take the top 2 channels
        top_rows = sorted_rows[:2]

        # Determine if any top channel has Δ2_med > Δ1_med
        cell_bad = False
        for r in top_rows:
            d1 = r.get("d1_med")
            d2 = r.get("d2_med")
            if d1 is not None and d2 is not None and d2 > d1:
                cell_bad = True
                break

        # Apply override
        if cell_bad:
            delta_dict[fluor] = ("bead", bead_delta[fluor])
        else:
            delta_dict[fluor] = ("cell", cell_delta[fluor])

    # Bundle raw deltas for QC
    raw_deltas = {
        "bead": bead_delta,
        "cell": cell_delta,
    }

    # 2. Write delta table
    write_delta_table(delta_dict, outdir / "delta_table.txt")

    # 3. Build channel order
    # PATCH 4: Sort channel_order for deterministic matrix output.
    # Previously depended on dict insertion order, which varied with fluor iteration order.
    seen: dict[str, None] = {}
    for source, rows in delta_dict.values():
        for r in rows:
            seen.setdefault(r["channel"], None)
    channel_order = sorted(seen.keys())

    # 4. Spectral vectors — unpack (source, rows) tuples
    spectral_vectors = {
        fluor: extract_spectral_vector(rows, channel_order)
        for fluor, (source, rows) in delta_dict.items()
    }
    fluor_order = sorted(spectral_vectors.keys())

    # 5. Mixing matrix
    M = build_mixing_matrix(spectral_vectors, fluor_order)

    # 6. Unmixing matrix
    U = build_unmixing_matrix(M)

    # 7. Write outputs
    pd.DataFrame(M, index=fluor_order, columns=channel_order).to_csv(
        outdir / "mixing_matrix.tsv", sep="\t"
    )
    pd.DataFrame(U, index=fluor_order, columns=channel_order).to_csv(
        outdir / "unmixing_matrix.tsv", sep="\t"
)
    (outdir / "fluor_order.txt").write_text("\n".join(fluor_order))
    (outdir / "channel_order.txt").write_text("\n".join(channel_order))

    # 8. QC PDF
    qc_pdf = write_qc_pdf(
        out_pdf=outdir / "unmixing_QC.pdf",
        structure=structure,
        mixing_matrix=M,
        unmixing_matrix=U,
        fluor_order=fluor_order,
        channel_order=channel_order,
        delta_table=delta_dict,
        raw_deltas=raw_deltas,
    )

    return {
        "delta_table":      outdir / "delta_table.txt",
        "mixing_matrix":    outdir / "mixing_matrix.tsv",
        "unmixing_matrix":  outdir / "unmixing_matrix.tsv",
        "fluor_order":      outdir / "fluor_order.txt",
        "channel_order":    outdir / "channel_order.txt",
        "qc_pdf":           qc_pdf,
    }


# ============================================================
# 6. QC PDF GENERATION
# ============================================================
def write_qc_pdf(
    out_pdf: Path,
    structure: ExperimentStructure,
    mixing_matrix: np.ndarray,
    unmixing_matrix: np.ndarray,
    fluor_order: list[str],
    channel_order: list[str],
    delta_table: dict[str, tuple[str, list[dict]]],
    raw_deltas: dict[str, dict[str, list[dict]]],
):
    """
    Clean, modern QC PDF generator.
    - Removes Time axis from spectral shape plots
    - Uses tight_layout()
    - Clears figures between pages
    - Produces readable, Cytek-style QC pages
    """

    with PdfPages(out_pdf) as pdf:

        # ============================================================
        # COVER PAGE
        # ============================================================
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(
            0.5, 0.8,
            "Spectral Unmixing QC Report",
            ha="center", va="center",
            fontsize=28, fontweight="bold"
        )
        ax.text(
            0.5, 0.6,
            f"{len(fluor_order)} fluors\n{len(channel_order)} channels",
            ha="center", va="center",
            fontsize=18
        )
        pdf.savefig(fig)
        plt.close(fig)

        # ============================================================
        # WARNINGS PAGE
        # ============================================================
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        if structure.has_warnings:
            ax.text(0.5, 0.9, "Warnings", ha="center", fontsize=24, fontweight="bold")
            y = 0.8
            for w in structure.warnings:
                ax.text(0.1, y, f"- {w}", fontsize=12)
                y -= 0.05
        else:
            ax.text(
                0.5, 0.5,
                "No warnings detected.\nAll controls appear complete.",
                ha="center", va="center",
                fontsize=18
            )

        pdf.savefig(fig)
        plt.close(fig)

        # ============================================================
        # IDENTITY MATRIX PAGE
        # ============================================================
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(unmixing_matrix @ mixing_matrix.T, cmap="viridis")
        ax.set_title("Identity Matrix (U @ Mᵀ)", fontsize=18)
        ax.set_xticks(range(len(fluor_order)))
        ax.set_yticks(range(len(fluor_order)))
        ax.set_xticklabels(fluor_order, rotation=90)
        ax.set_yticklabels(fluor_order)
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ============================================================
        # BEAD vs CELL Δ SUMMARY TABLE
        # ============================================================
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title("Bead vs Cell Δ Summary", fontsize=18, pad=20)

        bead_delta = raw_deltas.get("bead", {})
        cell_delta = raw_deltas.get("cell", {})

        def mean_or_none(rows, key):
            if rows is None:
                return None
            vals = [r.get(key) for r in rows if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        def fmt(val):
            return "NA" if val is None else f"{val:.1f}"

        table_data = [
            [
                "Fluor",
                "Cell Δ1_med", "Cell Δ1_p99",
                "Cell Δ2_med", "Cell Δ2_p99",
                "Bead Δ1_med", "Bead Δ1_p99",
                "Bead Δ2_med", "Bead Δ2_p99",
            ]
        ]
        row_meta = []

        all_fluors = sorted(set(bead_delta.keys()) | set(cell_delta.keys()))

        for fluor in all_fluors:
            cell_rows = cell_delta.get(fluor)
            bead_rows = bead_delta.get(fluor)

            cell_d1_med = mean_or_none(cell_rows, "d1_med")
            cell_d1_p99 = mean_or_none(cell_rows, "d1_p99")
            cell_d2_med = mean_or_none(cell_rows, "d2_med")
            cell_d2_p99 = mean_or_none(cell_rows, "d2_p99")

            bead_d1_med = mean_or_none(bead_rows, "d1_med")
            bead_d1_p99 = mean_or_none(bead_rows, "d1_p99")
            bead_d2_med = mean_or_none(bead_rows, "d2_med")
            bead_d2_p99 = mean_or_none(bead_rows, "d2_p99")

            # PATCH 5: Source comes directly from the tuple — no `is` comparison needed.
            used_source = delta_table[fluor][0] if fluor in delta_table else None

            row_meta.append({
                "fluor": fluor,
                "used": used_source,
                "cell_d1_med": cell_d1_med,
                "cell_d2_med": cell_d2_med,
                "bead_d1_med": bead_d1_med,
                "bead_d2_med": bead_d2_med,
            })

            table_data.append([
                fluor,
                fmt(cell_d1_med), fmt(cell_d1_p99),
                fmt(cell_d2_med), fmt(cell_d2_p99),
                fmt(bead_d1_med), fmt(bead_d1_p99),
                fmt(bead_d2_med), fmt(bead_d2_p99),
            ])

        table = ax.table(
            cellText=table_data,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        # Header bold
        for j in range(len(table_data[0])):
            table[(0, j)].get_text().set_fontweight("bold")


        # Style rows: bold used source, red when Δ2_med > Δ1_med
        for i, meta in enumerate(row_meta, start=1):
            used = meta["used"]

            if used == "cell":
                for j in range(1, 5):
                    table[(i, j)].get_text().set_fontweight("bold")
            elif used == "bead":
                for j in range(5, 9):
                    table[(i, j)].get_text().set_fontweight("bold")


            if meta["cell_d1_med"] is not None and meta["cell_d2_med"] is not None:
                if meta["cell_d2_med"] > meta["cell_d1_med"]:
                    table[(i, 3)].get_text().set_color("red")
            if meta["bead_d1_med"] is not None and meta["bead_d2_med"] is not None:
                if meta["bead_d2_med"] > meta["bead_d1_med"]:
                    table[(i, 7)].get_text().set_color("red")

        pdf.savefig(fig)
        plt.close(fig)

        # ============================================================
        # PER-FLUOR PAGES
        # ============================================================
        for fluor in fluor_order:
            # PATCH 5: Unpack (source, rows) tuple.
            _source, rows_final = delta_table[fluor]
            df_final = pd.DataFrame(rows_final)

            cell_rows = cell_delta.get(fluor)
            bead_rows = bead_delta.get(fluor)

            def rows_by_channel(rows):
                if rows is None:
                    return {}
                return {r["channel"]: r for r in rows}

            cell_map = rows_by_channel(cell_rows)
            bead_map = rows_by_channel(bead_rows)

            def build_curves(row_map):
                ss = []
                fmo = []
                un = []
                for ch in channel_order:
                    r = row_map.get(ch)
                    if r is None:
                        ss.append(np.nan)
                        fmo.append(np.nan)
                        un.append(np.nan)
                    else:
                        ss.append(r.get("ss_med"))
                        fmo.append(r.get("fmo_med"))
                        un.append(r.get("un_med"))
                return np.array(ss, float), np.array(fmo, float), np.array(un, float)

            cell_ss, cell_fmo, cell_un = build_curves(cell_map) if cell_map else (None, None, None)
            bead_ss, bead_fmo, bead_un = build_curves(bead_map) if bead_map else (None, None, None)

            # -----------------------------
            # 1. Spectral Shape (median)
            # -----------------------------
            fig, ax = plt.subplots(figsize=(10, 6))

            if cell_ss is not None and np.isfinite(cell_ss).any():
                ax.plot(channel_order, cell_ss, label="Cell SS median", color="blue")
            if cell_un is not None and np.isfinite(cell_un).any():
                ax.plot(channel_order, cell_un, label="Cell UN median", color="gray")
            if cell_fmo is not None and np.isfinite(cell_fmo).any():
                ax.plot(channel_order, cell_fmo, label="Cell FMO median", color="orange")

            if bead_ss is not None and np.isfinite(bead_ss).any():
                ax.plot(channel_order, bead_ss, label="Bead SS median", color="green")
            if bead_un is not None and np.isfinite(bead_un).any():
                ax.plot(
                    channel_order,
                    bead_un,
                    label="Bead UN median",
                    color="lightgray",
                    linestyle="--",
                )
            if bead_fmo is not None and np.isfinite(bead_fmo).any():
                ax.plot(channel_order, bead_fmo, label="Bead FMO median", color="red")

            ax.set_title(f"{fluor} – Spectral Shape (median)", fontsize=16)
            ax.set_xlabel("Detector")
            ax.set_ylabel("Intensity")
            ax.tick_params(axis="x", rotation=90)
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # -----------------------------
            # 2. Crosstalk After Unmixing
            # -----------------------------
            fig, ax = plt.subplots(figsize=(10, 6))

            # Row of U corresponds to fluor → contributions into channels
            unmixed_vec = unmixing_matrix[fluor_order.index(fluor), :]

            sns.barplot(
                x=channel_order,
                y=unmixed_vec,
                ax=ax,
                palette="viridis",
                legend=False
            )

            ax.set_title(f"{fluor} – Crosstalk After Unmixing", fontsize=16)
            ax.set_ylabel("Unmixed Intensity")
            ax.set_xlabel("Detector")
            ax.tick_params(axis="x", rotation=90)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


    return out_pdf


# ============================================================
# 7. CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified spectral unmixing builder.")
    parser.add_argument("experiment_root", help="Folder containing gated controls")
    parser.add_argument("--out", default="unmixing_output", help="Output folder")
    args = parser.parse_args()

    structure = organize(Path(args.experiment_root))
    results = run_unmixing(structure, Path(args.out))

    print("Unmixing complete.")
    for k, v in results.items():
        print(f"  {k}: {v}")