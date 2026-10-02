#!/usr/bin/env python3
"""
batch_unmix_GitHub.py — Apply a spectral unmixing matrix to a folder of FCS files.

Usage:
    python batch_unmix_GitHub.py \
        --matrix  results/unmixing/Exp1/unmixing_matrix.tsv \
        --input   analysis/Exp1/Exp1_cells \
        --output  results/unmixed/Exp1

FCS reader and writer are standalone pure-Python (no FCS library needed).
Output FCS files contain: scatter channels (FSC, SSC) + unmixed fluor columns.
Open them in Flow_View_GitHub.py to see the unmixed data and draw gates.
"""

import argparse
import glob
import os
import struct
import sys
import warnings
from pathlib import Path

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
    if len(parts) % 2 != 0:
        warnings.warn(
            f"TEXT segment has odd number of tokens ({len(parts)}); "
            "last token will be ignored — file may be malformed.",
            UserWarning,
        )
    text_dict = {}
    for i in range(0, len(parts) - 1, 2):
        key = parts[i].strip().lower()
        val = parts[i + 1].strip()
        text_dict[key] = val
    return text_dict


def read_fcs(path):
    with open(path, "rb") as f:
        raw = f.read()

    header = raw[:58].decode("ascii", errors="ignore")
    fcs_type = header[:6].strip()
    if not fcs_type.startswith("FCS"):
        raise ValueError(f"Not a valid FCS file (got header: {fcs_type!r})")

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

    if data_start == 0:
        for i in range(text_end + 1, len(raw)):
            if raw[i] not in b"0123456789.-+eE \t\r\n":
                data_start = i
                break

    par = int(meta("par", 0))
    tot = int(meta("tot", 0))

    data = raw[data_start:data_end + 1]

    dtype = str(meta("datatype", "I")).upper()
    byteord = str(meta("byteord", "1,2,3,4"))
    little_endian = byteord.startswith("1")

    if dtype == "I":
        bits_per_channel = [int(meta(f"p{i}b", 16)) for i in range(1, max(par, 1) + 1)]
        if par > 0 and len(set(bits_per_channel)) > 1:
            raise ValueError(f"Mixed per-channel bit depths: {bits_per_channel}")
        bits = bits_per_channel[0] if par > 0 else 32
        endian = "<" if little_endian else ">"
        if bits == 8:
            dt = np.dtype(f"{endian}u1")
        elif bits == 16:
            dt = np.dtype(f"{endian}u2")
        elif bits == 32:
            dt = np.dtype(f"{endian}u4")
        elif bits == 64:
            dt = np.dtype(f"{endian}u8")
        else:
            raise ValueError(f"Unsupported integer bit depth: {bits}")
    elif dtype == "F":
        dt = np.dtype("<f4") if little_endian else np.dtype(">f4")
    elif dtype == "D":
        dt = np.dtype("<f8") if little_endian else np.dtype(">f8")
    else:
        raise ValueError(f"Unsupported DATATYPE: {dtype}")

    arr = np.frombuffer(data, dtype=dt)

    if par == 0:
        print(f"⚠ {os.path.basename(path)} — $PAR missing, inferring from DATA segment")
        common_par_values = [40, 48, 50, 64, 67, 71, 73, 80]
        inferred = None
        for guess in common_par_values:
            if len(arr) % guess == 0:
                inferred = guess
                break
        if inferred is None:
            raise ValueError(f"Cannot infer PAR for {path} — file may be malformed.")
        par = inferred
        print(f"  → Inferred PAR = {par}")

    channel_names = []
    for i in range(1, par + 1):
        candidates = [
            meta(f"p{i}n"), meta(f"p{i}s"), meta(f"p{i}display"),
            meta(f"p{i}type"), meta(f"p{i}longname"), meta(f"p{i}shortname"),
        ]
        name = next((c for c in candidates if c and c.strip()), None)
        if not name:
            name = f"Ch{i}"
        channel_names.append(name)

    if tot > 0 and len(arr) == tot * par:
        events = arr.reshape((tot, par))
    else:
        n = len(arr) // par
        if n == 0:
            raise ValueError("DATA segment too small to contain any events.")
        events = arr[: n * par].reshape((n, par))

    return events, channel_names, t


def load_fcs_as_dataframe(path):
    events, channel_names, _ = read_fcs(path)
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
        text_end = text_start + len(text_bytes) - 1
        data_start = text_end + 1
        data_end = data_start + len(data_bytes) - 1

        if data_end > 99_999_999:
            text_bytes = build_text([
                ("$BEGINDATA", str(data_start)),
                ("$ENDDATA", str(data_end)),
            ])
            hdr_data_start = "       0"
            hdr_data_end = "       0"
        else:
            hdr_data_start = f"{data_start:>8}"
            hdr_data_end = f"{data_end:>8}"

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

    print(f"  ✓ Wrote FCS 3.0 file : {filename}")
    print(f"    Events : {n_events:,}  |  Channels : {n_channels}")


# ─────────────────────────────────────────────────────────────
# UNMIXING MATRIX LOADER
# ─────────────────────────────────────────────────────────────

def load_matrix(path):
    """
    unmixer.py writes rows=fluors, cols=detectors.
    We need (detectors x fluors) for:  unmixed = detector_data @ U
    So we transpose on load.
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    print(f"  Matrix loaded: {df.shape[0]} rows × {df.shape[1]} cols  (fluors × detectors)")
    U = df.T   # → (detectors × fluors)
    return U.values.astype(np.float64), list(U.index), list(U.columns)


# ─────────────────────────────────────────────────────────────
# APPLY UNMIXING
# ─────────────────────────────────────────────────────────────

def apply_unmixing(df_raw, U, detectors, fluors):
    scatter_cols = [c for c in df_raw.columns if c not in detectors]

    missing = [d for d in detectors if d not in df_raw.columns]
    if missing:
        warnings.warn(
            f"{len(missing)} detector(s) missing from FCS — filled with 0:\n  {missing}",
            UserWarning,
        )

    det_matrix = np.zeros((len(df_raw), len(detectors)), dtype=np.float64)
    for j, ch in enumerate(detectors):
        if ch in df_raw.columns:
            det_matrix[:, j] = df_raw[ch].values

    unmixed = det_matrix @ U   # (n_events × n_fluors)

    df_fluors = pd.DataFrame(unmixed, columns=fluors, index=df_raw.index)
    df_out = pd.concat([
        df_raw[scatter_cols].reset_index(drop=True),
        df_fluors.reset_index(drop=True)
    ], axis=1)
    return df_out


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch spectral unmixing for gated FCS files.")
    parser.add_argument("--matrix",  required=True, help="Path to unmixing_matrix.tsv")
    parser.add_argument("--input",   required=True, help="Folder of *.fcs files")
    parser.add_argument("--output",  required=True, help="Output folder for *_unmixed.fcs")
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    input_dir   = Path(args.input)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading unmixing matrix: {matrix_path}")
    U, detectors, fluors = load_matrix(matrix_path)
    print(f"  → U after transpose: {U.shape}  ({len(detectors)} detectors × {len(fluors)} fluors)")
    print(f"  Fluors: {fluors}\n")

    fcs_files = sorted(input_dir.glob("*.fcs"))
    if not fcs_files:
        print(f"No *.fcs files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(fcs_files)} gated FCS files → unmixing...\n")

    ok, skipped = 0, 0
    for fcs_path in fcs_files:
        out_name = fcs_path.name.replace("_gated.fcs", "_unmixed.fcs")
        out_path = output_dir / out_name

        try:
            df_raw = load_fcs_as_dataframe(fcs_path)

            if len(df_raw) == 0:
                print(f"  ⚠  {fcs_path.name} — 0 events, skipping")
                skipped += 1
                continue

            df_out = apply_unmixing(df_raw, U, detectors, fluors)
            write_fcs_3_0(str(out_path), df_out)
            ok += 1

        except Exception as e:
            print(f"  ✗  {fcs_path.name} — ERROR: {e}")
            skipped += 1

    print(f"\nDone.  {ok} unmixed,  {skipped} skipped.")
    print(f"Output → {output_dir}")


if __name__ == "__main__":
    main()
