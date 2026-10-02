#fcs reader.py — Pure-Python FCS reader (linear only, FCS 3.0/3.1/3.2 compliant)
import numpy as np
import struct
import re
import os
import pandas as pd

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