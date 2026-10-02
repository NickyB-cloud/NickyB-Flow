#!/usr/bin/env python3
"""
Apply_sequential_gates_GitHub.py
Applies a single gate (threshold OR polygon hull) to a folder of FCS files.
Run once per gate step (live/dead -> CD11c+ -> BMDC/BMDM polygons, etc.).

Usage examples:
    # Threshold gate (1D)
    python Apply_sequential_gates_GitHub.py \
        --input  results/unmixed/Exp1 \
        --output analysis/Exp1/gated/Exp1_live \
        --threshold analysis/Exp1/gates/viability_gate_threshold.json

    # Polygon gate (2D)
    python Apply_sequential_gates_GitHub.py \
        --input   analysis/Exp1/gated/Exp1_cd11cpos \
        --output  analysis/Exp1/gated/Exp1_BMDCs \
        --hull    analysis/Exp1/gates/mhcii_cd11b_gate_hull.txt \
        --hull-x  PERCP \
        --hull-y  APC
    (--hull-x/--hull-y must be the x/y channels the polygon was drawn on;
     here x = MHC-II on PERCP, y = CD11b on APC)
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


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
    header    = raw[:58].decode("ascii", errors="ignore")
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
# FCS WRITER — standalone pure-Python implementation
# ─────────────────────────────────────────────────────────────

def write_fcs_3_0(filename, df):
    data = df.values.astype("float32")
    n_events, n_channels = data.shape
    channels = df.columns.tolist()
    delim = "|"

    def sanitize(name):
        return str(name).replace("|", "/")

    text_items = []
    text_items.append(("beginanalysis", "0"))
    text_items.append(("endanalysis", "0"))
    text_items.append(("byteord", "1,2,3,4"))
    text_items.append(("datatype", "F"))
    text_items.append(("mode", "L"))
    text_items.append(("nextdata", "0"))
    text_items.append(("tot", str(n_events)))
    text_items.append(("par", str(n_channels)))

    for i, ch in enumerate(channels, start=1):
        safe = sanitize(ch)
        ch_max = df[ch].max()
        rng = str(int(ch_max) + 1) if np.isfinite(ch_max) else "4194304"
        text_items.append((f"p{i}b", "32"))
        text_items.append((f"p{i}e", "0,0"))
        text_items.append((f"p{i}n", safe))
        text_items.append((f"p{i}r", rng))
        text_items.append((f"p{i}display", "LOG"))
        text_items.append((f"p{i}type", "Raw_Fluorescence"))
        text_items.append((f"p{i}v", "0"))

    def build_text(extra_items=None):
        items = text_items.copy()
        if extra_items:
            items.extend(extra_items)
        txt = "|" + "|".join(f"{k}|{v}" for k, v in items) + "|"
        return txt.encode("ascii")

    data_bytes = struct.pack(f"<{data.size}f", *data.flatten())
    text_start = 256
    text_bytes = build_text()

    for _ in range(2):
        text_end   = text_start + len(text_bytes) - 1
        data_start = text_end + 1
        data_end   = data_start + len(data_bytes) - 1

        if data_end > 99_999_999:
            text_bytes = build_text([
                ("$BEGINDATA", str(data_start)),
                ("$ENDDATA",   str(data_end)),
            ])
            hdr_data_start = "       0"
            hdr_data_end   = "       0"
        else:
            hdr_data_start = f"{data_start:>8}"
            hdr_data_end   = f"{data_end:>8}"

    header = (
        b"FCS3.0    "
        + f"{text_start:>8}".encode("ascii")
        + f"{text_end:>8}".encode("ascii")
        + hdr_data_start.encode("ascii")
        + hdr_data_end.encode("ascii")
        + f"{'0':>8}".encode("ascii")
        + f"{'0':>8}".encode("ascii")
        + b" " * (256 - 58)
    )

    with open(filename, "wb") as fh:
        fh.write(header)
        fh.write(text_bytes)
        fh.write(data_bytes)

    print(f"  ✓ Wrote {os.path.basename(filename)}  ({n_events:,} events)")


# ─────────────────────────────────────────────────────────────
# GATE LOADERS
# ─────────────────────────────────────────────────────────────

def load_threshold(json_path):
    with open(json_path) as f:
        return json.load(f)
    # Returns: {channel, value, direction, transform, cofactor}


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
# GATE APPLICATION
# ─────────────────────────────────────────────────────────────

def arcsinh(x, cofactor=150):
    return np.arcsinh(np.asarray(x, dtype=float) / cofactor)


def apply_threshold(df, gate_spec):
    channel   = gate_spec["channel"]
    value     = gate_spec["value"]
    direction = gate_spec["direction"]
    transform = gate_spec.get("transform", "arcsinh")
    cofactor  = gate_spec.get("cofactor", 150)

    if channel not in df.columns:
        raise ValueError(f"Channel '{channel}' not found in FCS. "
                         f"Available: {list(df.columns)}")

    if transform == "arcsinh":
        vals = arcsinh(df[channel].values, cofactor)
    else:
        vals = df[channel].values

    if direction == "below":
        mask = vals <= value
    else:
        mask = vals >= value

    return df[mask]


def apply_hull(df, coords, x_col, y_col, transform="arcsinh", cofactor=150):
    if x_col not in df.columns:
        raise ValueError(f"X channel '{x_col}' not found. Available: {list(df.columns)}")
    if y_col not in df.columns:
        raise ValueError(f"Y channel '{y_col}' not found. Available: {list(df.columns)}")

    if transform == "arcsinh":
        x_vals = arcsinh(df[x_col].values, cofactor)
        y_vals = arcsinh(df[y_col].values, cofactor)
    else:
        x_vals = df[x_col].values
        y_vals = df[y_col].values

    poly = Polygon(coords)
    mask = np.array([
        poly.contains(Point(x, y))
        for x, y in zip(x_vals, y_vals)
    ])
    return df[mask]


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apply one gate (threshold or polygon) to a folder of FCS files."
    )
    parser.add_argument("--input",     required=True,
                        help="Input folder of FCS files")
    parser.add_argument("--output",    required=True,
                        help="Output folder for gated FCS files")
    parser.add_argument("--threshold", default=None,
                        help="Path to threshold .json (1D gate)")
    parser.add_argument("--hull",      default=None,
                        help="Path to hull .txt (2D polygon gate)")
    parser.add_argument("--hull-x",    default=None,
                        help="X channel name for hull gate (e.g. APC)")
    parser.add_argument("--hull-y",    default=None,
                        help="Y channel name for hull gate (e.g. PERCP)")
    parser.add_argument("--transform", default="arcsinh",
                        choices=["arcsinh", "linear"],
                        help="Transform to apply before polygon gate (default: arcsinh)")
    args = parser.parse_args()

    if not args.threshold and not args.hull:
        print("❌  Provide either --threshold or --hull")
        sys.exit(1)

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load gate spec
    if args.threshold:
        gate_type = "threshold"
        gate_spec = load_threshold(args.threshold)
        print(f"\nThreshold gate: {gate_spec['channel']} "
              f"{gate_spec['direction']} {gate_spec['value']:.3f} "
              f"(arcsinh, cofactor={gate_spec.get('cofactor', 150)})")
    else:
        gate_type = "hull"
        coords    = load_hull(args.hull)
        print(f"\nPolygon gate: {len(coords)} vertices | "
              f"X={args.hull_x} Y={args.hull_y}")

    # Find FCS files
    fcs_files = sorted(input_dir.rglob("*.fcs"))
    if not fcs_files:
        print(f"❌  No .fcs files found in {input_dir}")
        sys.exit(1)

    print(f"Processing {len(fcs_files)} FCS files → {output_dir}\n")

    ok, skipped = 0, 0
    n_before_total, n_after_total = 0, 0

    for fcs_path in fcs_files:
        out_path = output_dir / fcs_path.name

        try:
            df = read_fcs(fcs_path)
            n_before = len(df)

            if n_before == 0:
                print(f"  ⚠  {fcs_path.name} — 0 events, skipping")
                skipped += 1
                continue

            if gate_type == "threshold":
                df_gated = apply_threshold(df, gate_spec)
            else:
                df_gated = apply_hull(df, coords, args.hull_x, args.hull_y,
                                      transform=args.transform)

            n_after = len(df_gated)
            pct     = 100 * n_after / n_before if n_before > 0 else 0

            if n_after == 0:
                print(f"  ⚠  {fcs_path.name} — 0 events after gate "
                      f"(started with {n_before:,}) — skipping write")
                skipped += 1
                continue

            write_fcs_3_0(str(out_path), df_gated)
            print(f"     {n_before:,} → {n_after:,} ({pct:.1f}% retained)")

            n_before_total += n_before
            n_after_total  += n_after
            ok += 1

        except Exception as e:
            print(f"  ✗  {fcs_path.name} — ERROR: {e}")
            skipped += 1

    pct_total = 100 * n_after_total / n_before_total if n_before_total > 0 else 0
    print(f"\nDone.  {ok} gated,  {skipped} skipped.")
    print(f"Total: {n_before_total:,} → {n_after_total:,} cells "
          f"({pct_total:.1f}% retained)")
    print(f"Output → {output_dir}")


if __name__ == "__main__":
    main()