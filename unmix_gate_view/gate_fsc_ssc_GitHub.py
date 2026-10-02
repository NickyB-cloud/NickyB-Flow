# FSC/SSC Gating Script v4
import glob
import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPoint
import hdbscan
import struct
import argparse
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# EDIT FOR YOUR DATA
# Polygon (FSC-A vs SSC-A) used by --gate boolean. One "(x, y)" pair per
# line -- the same format this script writes as *_hull.txt when run with
# --gate hdbscan, so a typical workflow is: run hdbscan on an unstained
# control, check the PNG, then point this at the hull it saved.
# Currently hdbscan is optimized for beads and not cells.
# ------------------------------------------------------------
BOOLEAN_HULL_PATH = "path/to/your/unstained_control_hull.txt"


# ------------------------------------------------------------
# PURE-PYTHON FCS READER (FCS 3.0/3.1/3.2 compliant, linear only)
# ------------------------------------------------------------


def parse_text_segment(text_block):
    """
    Parse the TEXT segment of an FCS file.
    Handles arbitrary delimiters, missing keys, lowercase keys,
    and Cytek-style malformed TEXT blocks.
    """
    if not text_block:
        return {}

    # First character is the delimiter
    delim = text_block[0]

    # Handle escaped delimiters (FCS 3.1 spec: doubled delimiter = literal)
    placeholder = "\x00"
    text_block = text_block.replace(delim * 2, placeholder)
    parts = text_block.strip(delim).split(delim)
    parts = [p.replace(placeholder, delim) for p in parts]

    # Warn if odd number of parts (last value would be silently dropped)
    if len(parts) % 2 != 0:
        import warnings
        warnings.warn(
            f"TEXT segment has odd number of tokens ({len(parts)}); "
            "last token will be ignored — file may be malformed.",
            UserWarning,
        )

    # Convert to dict: key1, val1, key2, val2, ...
    text_dict = {}
    for i in range(0, len(parts) - 1, 2):
        key = parts[i].strip().lower()
        val = parts[i + 1].strip()
        text_dict[key] = val

    return text_dict

def read_fcs(path):
    with open(path, "rb") as f:
        raw = f.read()

    # ------------------------------
    # HEADER
    # ------------------------------
    header = raw[:58].decode("ascii", errors="ignore")
    fcs_type = header[:6].strip()
    if not fcs_type.startswith("FCS"):
        raise ValueError(f"Not a valid FCS file (got header: {fcs_type!r})")

    text_start = int(header[10:18].strip() or 0)
    text_end   = int(header[18:26].strip() or 0)
    data_start = int(header[26:34].strip() or 0)
    data_end   = int(header[34:42].strip() or 0)

    # ------------------------------
    # TEXT SEGMENT
    # ------------------------------
    text_block = raw[text_start:text_end + 1].decode("ascii", errors="ignore")
    t = parse_text_segment(text_block)

    # Helper to read both bare and $-prefixed keys
    def meta(key, default=None):
        return t.get(key, t.get(f"${key}", default))

    # TEXT overrides header offsets
    data_start = int(meta("begindata", data_start))
    data_end   = int(meta("enddata",   data_end))

    # Fallback: scan for first binary byte
    if data_start == 0:
        for i in range(text_end + 1, len(raw)):
            if raw[i] not in b"0123456789.-+eE \t\r\n":
                data_start = i
                break

    # ------------------------------
    # Extract PAR and TOT (corrected)
    # ------------------------------
    par = int(meta("par", 0))
    tot = int(meta("tot", 0))

    # ------------------------------
    # SLICE DATA BLOCK FIRST
    # ------------------------------
    data = raw[data_start:data_end + 1]

    # ------------------------------
    # DETERMINE NUMPY DTYPE (corrected)
    # ------------------------------
    dtype = str(meta("datatype", "I")).upper()
    byteord = str(meta("byteord", "1,2,3,4"))
    little_endian = byteord.startswith("1")

    if dtype == "I":
        bits_per_channel = [int(meta(f"p{i}b", 16)) for i in range(1, max(par,1) + 1)]
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

    # ------------------------------
    # CONVERT TO NUMPY ARRAY
    # ------------------------------
    arr = np.frombuffer(data, dtype=dt)

    # ------------------------------
    # NOW SAFELY INFER PAR IF MISSING
    # ------------------------------
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

    # ------------------------------
    # BUILD CHANNEL NAMES (Aurora-safe)
    # ------------------------------
    channel_names = []
    for i in range(1, par + 1):
        candidates = [
            meta(f"p{i}n"),
            meta(f"p{i}s"),
            meta(f"p{i}display"),
            meta(f"p{i}type"),
            meta(f"p{i}longname"),
            meta(f"p{i}shortname"),
        ]
        name = next((c for c in candidates if c and c.strip()), None)
        if not name:
            name = f"Ch{i}"
        channel_names.append(name)

    # ------------------------------
    # RESHAPE EVENTS
    # ------------------------------
    if tot > 0 and len(arr) == tot * par:
        events = arr.reshape((tot, par))
    else:
        n = len(arr) // par
        if n == 0:
            raise ValueError("DATA segment too small to contain any events.")
        events = arr[: n * par].reshape((n, par))

    return events, channel_names, t



def load_fcs_as_dataframe(path):
    """
    High-level wrapper: returns a pandas DataFrame with linear values.
    """
    events, channel_names, _ = read_fcs(path)
    return pd.DataFrame(events, columns=channel_names)


# ---------------------------------------------------------------------------
# FCS 3.0 Writer
# ---------------------------------------------------------------------------
def write_fcs_3_0(filename, df):
    data = df.values.astype("float32")
    n_events, n_channels = data.shape
    channels = df.columns.tolist()
    delim = "|"

    def sanitize(name):
        return str(name).replace("|", "/")

    # ---- Build TEXT in strict universal ordering ----
    # ---- Build TEXT in Cytek Aurora ordering ----
    text_items = []

    # Global keys (Cytek order)
    text_items.append(("beginanalysis", "0"))
    text_items.append(("endanalysis", "0"))
    text_items.append(("byteord", "1,2,3,4"))
    text_items.append(("datatype", "F"))
    text_items.append(("mode", "L"))
    text_items.append(("nextdata", "0"))
    text_items.append(("tot", str(n_events)))
    text_items.append(("par", str(n_channels)))

    # Per-parameter keys (Cytek order)
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

    # Build TEXT block
    def build_text(extra_items=None):
        items = text_items.copy()
        if extra_items:
            items.extend(extra_items)
        txt = "|" + "|".join(f"{k}|{v}" for k, v in items) + "|"
        return txt.encode("ascii")


    # ---- DATA block ----
    data_bytes = struct.pack(f"<{data.size}f", *data.flatten())

    # ---- Two-pass offset resolution ----
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

    # ---- Header ----
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

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# FSC/SSC Gating
#----------------------------------------------------------------------------

#-----------------------------
# Core gating entry points
#-----------------------------

def gate_folder(input_folder, output_folder, gate_fn):
    os.makedirs(output_folder, exist_ok=True)
    fcs_files = glob.glob(os.path.join(input_folder, "*.fcs"))

    for f in fcs_files:
        fname = os.path.basename(f)
        print(f"Gating {fname}...")

        # --- SAFE LOAD WRAPPER ---
        try:
            df = load_fcs_as_dataframe(f)
        except Exception as e:
            print(f"⚠ Skipping unreadable file: {fname}")
            print(f"  Reason: {e}")
            continue
        # --------------------------

        # Apply gate
        result = gate_fn(df)

        if isinstance(result, tuple):
            df_gated, hull = result

            print("  HDBSCAN polygon hull:")
            for x, y in hull:
                print(f"    ({x:.1f}, {y:.1f})")

            hull_path = os.path.join(
                output_folder,
                fname.replace(".fcs", "_hull.txt")
            )
            save_polygon_txt(hull_path, hull)
        else:
            df_gated = result
            hull = []

        # Write gated FCS
        out_path = os.path.join(
            output_folder,
            fname.replace(".fcs", "_gated.fcs")
        )
        write_fcs_3_0(out_path, df_gated)
        print(f"  ✓ {os.path.basename(out_path)}  ({len(df_gated):,} events)")

        # --- Gate PNG (visual confirmation) ---
        png_path = os.path.join(
            output_folder,
            fname.replace(".fcs", "_gate.png")
        )

        if args.gate == "boolean":
            hull_path = BOOLEAN_HULL_PATH
            gate_polygon = clean_hull(load_polygon_txt(hull_path))
        else:
            gate_polygon = hull

        plot_gate(df, gate_polygon, png_path)




def gate_single_file(fcs_path, output_folder, gate_fn):
    os.makedirs(output_folder, exist_ok=True)
    print(f"Gating {os.path.basename(fcs_path)}...")

    # Load full, ungated data (for comparison plot)
    df = load_fcs_as_dataframe(fcs_path)

    # Run selected gate (HDBSCAN or boolean, etc.)
    result = gate_fn(df)

    if isinstance(result, tuple):
        df_gated, hull = result

        # Clean hull (remove duplicate closing point, sort, tuples)
        hull = clean_hull(hull)

        print("  HDBSCAN polygon hull:")
        for x, y in hull:
            print(f"    ({x:.1f}, {y:.1f})")

        # Save hull to .txt
        hull_path = os.path.join(
            output_folder,
            os.path.basename(fcs_path).replace(".fcs", "_hull.txt")
        )
        save_polygon_txt(hull_path, hull)
    else:
        df_gated = result
        hull = []  # no HDBSCAN hull available

    # Write gated FCS
    out_path = os.path.join(
        output_folder,
        os.path.basename(fcs_path).replace(".fcs", "_gated.fcs")
    )
    write_fcs_3_0(out_path, df_gated)
    print(f"  ✓ {os.path.basename(out_path)}  ({len(df_gated):,} events)")

    # Comparison PNG (boolean vs HDBSCAN)
    # --- Gate PNG ---
    png_path = os.path.join(
        output_folder,
        os.path.basename(fcs_path).replace(".fcs", "_gate.png")
    )

    # Determine which polygon to plot
    if args.gate == "boolean":
        hull_path = BOOLEAN_HULL_PATH
        gate_polygon = clean_hull(load_polygon_txt(hull_path))
    else:
        gate_polygon = hull

    plot_gate(df, gate_polygon, png_path)





#--------------------------------------
# Boolean polygon gate (FSC-A vs SSC-A)
#--------------------------------------

def load_polygon_txt(path):
    coords = []
    with open(path) as f:
        for line in f:
            coords.append(eval(line.strip()))

    if len(coords) > 2 and coords[0] == coords[-1]:
        coords = coords[:-1]

    return coords


def order_hull_points(points):
    """
    Sort hull points into polygon order by angle around centroid.
    """
    pts = np.array(points, dtype=float)
    cx, cy = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(angles)
    return pts[order]

def clean_hull(hull):
    if len(hull) > 2 and hull[0] == hull[-1]:
        hull = hull[:-1]
    hull = order_hull_points(hull)
    hull = [(float(x), float(y)) for x, y in hull]
    return hull

def gate_boolean_fsc_ssc(df, fsc="FSC-A", ssc="SSC-A"):
    hull_path = BOOLEAN_HULL_PATH
    raw_points = load_polygon_txt(hull_path)
    ordered = clean_hull(raw_points)

    gate = Polygon(ordered)
    mask = [gate.contains(Point(x, y)) for x, y in zip(df[fsc], df[ssc])]
    return df[np.array(mask)]


#---------------------------------
# HDBSCAN gating with convex hull
#---------------------------------

def compute_convex_hull(df, fsc="FSC-A", ssc="SSC-A"):
    pts = df[[fsc, ssc]].to_numpy()
    if len(pts) < 3:
        return []

    hull_poly = MultiPoint(pts).convex_hull
    x, y = hull_poly.exterior.coords.xy
    return list(zip(x, y))


def prefilter_debris(df, fsc="FSC-A", ssc="SSC-A"):
    df = df[df[fsc] > 5e4]
    df = df[df[ssc] > 5e4]
    df = df[df[fsc] < 4e6]
    df = df[df[ssc] < 4e6]
    return df


def gate_hdbscan_fsc_ssc(df, fsc="FSC-A", ssc="SSC-A"):
    """
    Automatically gate the main population using HDBSCAN on FSC-A/SSC-A.
    Returns (df_gated, hull_vertices)
    """

    # For beads, prefiltering with hard FSC/SSC cutoffs can kill everything.
    # So we skip prefilter_debris here and let HDBSCAN see all events.
    X = df[[fsc, ssc]].to_numpy()

    if len(X) < 10:
        print("⚠ Not enough events for HDBSCAN — returning original data")
        return df, []

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=200,   # smaller than 500, better for beads
        min_samples=20,
    ).fit(X)

    labels = clusterer.labels_
    valid = labels[labels >= 0]

    if len(valid) == 0:
        print("⚠ HDBSCAN found no clusters — returning original data")
        return df, []

    unique, counts = np.unique(valid, return_counts=True)
    main_cluster = unique[np.argmax(counts)]

    mask = labels == main_cluster
    df_gated = df[mask]

    hull = compute_convex_hull(df_gated, fsc=fsc, ssc=ssc)

    if len(hull) > 2 and hull[0] == hull[-1]:
        hull = hull[:-1]

    hull = order_hull_points(hull)
    hull = [(float(x), float(y)) for x, y in hull]

    return df_gated, hull



def save_polygon_txt(path, hull):
    with open(path, "w") as f:
        for x, y in hull:
            f.write(f"({x}, {y})\n")


#----------------------------------------------------------------------------
# Comparison of boolean vs HDBSCAN gating
#----------------------------------------------------------------------------

def plot_gate(df, gate_polygon, out_png, fsc="FSC-A", ssc="SSC-A"):
    plt.figure(figsize=(8, 8))

    # Raw data
    plt.scatter(df[fsc], df[ssc], s=1, alpha=0.2, color="gray",
                label="All events")

    # Gate polygon (if provided)
    if gate_polygon is not None and len(gate_polygon) > 0:
        gx, gy = zip(*gate_polygon)
        gx = list(gx) + [gx[0]]
        gy = list(gy) + [gy[0]]
        plt.plot(gx, gy, "g-", linewidth=2, label="Gate Used")

    plt.xlabel(fsc)
    plt.ylabel(ssc)
    plt.legend()
    plt.title("Applied Gate")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"  ✓ Saved gate PNG → {out_png}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Batch gate FCS files.")
    parser.add_argument("--gate", choices=["boolean", "hdbscan"], default="boolean")
    parser.add_argument("--input", help="Folder containing FCS files.")
    parser.add_argument("--output", required=True, help="Folder to write gated FCS files.")
    parser.add_argument("--file", help="Gate a single FCS file instead of a folder.")

    args = parser.parse_args()

    # Choose gating function
    gate_fn = gate_boolean_fsc_ssc if args.gate == "boolean" else gate_hdbscan_fsc_ssc

    # --- SINGLE FILE MODE ---
    if args.file:
        gate_single_file(
            fcs_path=args.file,
            output_folder=args.output,
            gate_fn=gate_fn
        )
        exit(0)

    # --- FOLDER MODE ---
    if not args.input:
        raise ValueError("You must provide either --file or --input")

    gate_folder(
        input_folder=args.input,
        output_folder=args.output,
        gate_fn=gate_fn
    )
