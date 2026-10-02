#!/usr/bin/env python3
"""
Flow_View_GitHub.py — Interactive gating interface for spectral flow cytometry
(originally Dash_v4.py).

Usage:
    cd <your project folder>
    python Flow_View_GitHub.py

Then open http://127.0.0.1:8050 in your browser, paste the path to an
.fcs file (optionally an FMO control for comparison) and draw gates.

Gates you can draw and save:
    2D polygon gate  → saves as hull .txt  (one "(x, y)" pair per line)
    1D threshold     → saves as threshold .json
Both formats are read by Apply_sequential_gates_GitHub.py (--hull / --threshold).

Example gating sequence (BMDC activation panel):
    1. FSC vs SSC        → cells (polygon)
    2. Viability dye (1D) → live-cell threshold
    3. CD11c (1D or 2D)  → CD11c+ gate
    4. MHC-II vs CD11b   → dendritic-cell vs macrophage polygons
"""

import json
import os
import struct
import warnings
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# FCS READER — standalone pure-Python reader (no FCS library needed)
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
    header = raw[:58].decode("ascii", errors="ignore")
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
# POLYGON GATING — same point-in-polygon logic as Apply_sequential_gates_GitHub.py
# ─────────────────────────────────────────────────────────────

def load_polygon_txt(path):
    coords = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                coords.append(eval(line))
    if len(coords) > 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return coords


def save_polygon_txt(path, coords):
    with open(path, "w") as f:
        for pt in coords:
            f.write(f"({pt[0]}, {pt[1]})\n")
    print(f"  ✓ Saved hull → {path}")


def apply_polygon_gate(df, coords, x_col, y_col):
    from shapely.geometry import Point, Polygon
    if len(coords) < 3:
        return df
    poly = Polygon(coords)
    mask = [poly.contains(Point(x, y))
            for x, y in zip(df[x_col].values, df[y_col].values)]
    return df[np.array(mask)]


def apply_threshold_gate(df, col, value, direction="below"):
    """direction: 'below' = keep cells below threshold (live), 'above' = keep above (positive)"""
    if direction == "below":
        return df[df[col] <= value]
    else:
        return df[df[col] >= value]


# ─────────────────────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────────────────────

def arcsinh(x, cofactor=150):
    return np.arcsinh(np.asarray(x, dtype=float) / cofactor)


def apply_transform(series, transform, cofactor=150):
    vals = series.values.astype(float)
    if transform == "arcsinh":
        return arcsinh(vals, cofactor)
    return vals  # linear


# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────

app = dash.Dash(__name__)
server = app.server

# Global data cache
CACHE = {
    "df_raw":      None,
    "df_gated":    None,
    "fcs_path":    None,
    "channels":    [],
    "poly_points": [],   # in-progress polygon clicks
}

# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────

PANEL_STYLE = {
    "display": "inline-block",
    "verticalAlign": "top",
    "padding": "15px",
    "border": "1px solid #ddd",
    "borderRadius": "6px",
    "margin": "8px",
    "backgroundColor": "#fafafa",
}

app.layout = html.Div([

    html.H2("Flow View — Interactive Gating",
            style={"textAlign": "center", "color": "#0E4D92", "marginBottom": "5px"}),
    html.P("Draw polygon gates (2D) or threshold gates (1D) and save as hull .txt or threshold .json",
           style={"textAlign": "center", "color": "#555", "marginTop": "0"}),

    html.Hr(),

    # ── TOP ROW: file loader + channel selector ──────────────
    html.Div([

        html.Div([
            html.H4("1. Load FCS"),
            html.Label("FCS file path:"),
            dcc.Input(id="fcs-path", type="text",
                      placeholder="/path/to/file_unmixed.fcs",
                      style={"width": "420px", "marginBottom": "8px",
                             "display": "block"}),
            html.Label("Optional: overlay FCS (e.g. FMO control):"),
            dcc.Input(id="fcs-overlay-path", type="text",
                      placeholder="/path/to/fmo_control.fcs",
                      style={"width": "420px", "marginBottom": "8px",
                             "display": "block"}),
            dcc.Checklist(
                id="use-pooled",
                options=[{"label": " Pool all FCS files in same folder (good for DMSO/OVA controls)", "value": "yes"}],
                value=[],
                style={"marginBottom": "8px", "fontSize": "12px"},
            ),
            html.Button("Load", id="load-btn",
                        style={"backgroundColor": "#0E4D92", "color": "white",
                               "border": "none", "padding": "8px 20px",
                               "borderRadius": "4px", "cursor": "pointer"}),
            html.Div(id="load-status",
                     style={"marginTop": "8px", "color": "#555", "fontSize": "13px"}),
        ], style=PANEL_STYLE),

        html.Div([
            html.H4("2. Axes + Transform"),
            html.Label("X axis:"),
            dcc.Dropdown(id="x-channel", style={"width": "260px", "marginBottom": "8px"}),
            html.Label("Y axis (leave blank for 1D histogram):"),
            dcc.Dropdown(id="y-channel", style={"width": "260px", "marginBottom": "8px"}),
            html.Label("Transform:"),
            dcc.RadioItems(
                id="transform",
                options=[
                    {"label": "Linear", "value": "linear"},
                    {"label": "arcsinh (÷150)", "value": "arcsinh"},
                ],
                value="arcsinh",
                inline=True,
                style={"marginBottom": "8px"},
            ),
            html.Label("Plot type (2D only):"),
            dcc.RadioItems(
                id="plot-type",
                options=[
                    {"label": "Scatter", "value": "scatter"},
                    {"label": "Density", "value": "density"},
                ],
                value="density",
                inline=True,
            ),
            html.Div([
                html.Label("Max events to display:"),
                dcc.Slider(id="subsample", min=1000, max=50000,
                           step=1000, value=10000,
                           marks={1000: "1k", 10000: "10k",
                                  25000: "25k", 50000: "50k"},
                           tooltip={"placement": "bottom"}),
            ], style={"marginTop": "10px"}),
        ], style=PANEL_STYLE),

        html.Div([
            html.H4("3. Gate Mode"),
            dcc.RadioItems(
                id="gate-mode",
                options=[
                    {"label": "2D Polygon (click to add vertices)", "value": "polygon"},
                    {"label": "1D Threshold (click on histogram)",   "value": "threshold"},
                ],
                value="polygon",
                style={"marginBottom": "12px"},
            ),
            html.Label("Threshold direction (1D only):"),
            dcc.RadioItems(
                id="threshold-direction",
                options=[
                    {"label": "Keep BELOW (e.g. viability — live = dye low)", "value": "below"},
                    {"label": "Keep ABOVE (e.g. CD11c+ = marker hi)",          "value": "above"},
                ],
                value="above",
                style={"marginBottom": "12px"},
            ),
            html.Div(id="polygon-status",
                     style={"fontSize": "13px", "color": "#555",
                            "minHeight": "40px", "marginBottom": "8px"}),
            html.Button("Undo last point", id="undo-btn",
                        style={"marginRight": "8px", "marginBottom": "8px"}),
            html.Button("Clear gate", id="clear-btn",
                        style={"marginRight": "8px", "marginBottom": "8px"}),
            html.Button("Apply gate to data", id="apply-gate-btn",
                        style={"backgroundColor": "#2ca02c", "color": "white",
                               "border": "none", "padding": "6px 14px",
                               "borderRadius": "4px", "cursor": "pointer",
                               "marginBottom": "8px"}),
            html.Div(id="gate-stats",
                     style={"fontSize": "13px", "color": "#333",
                            "fontWeight": "bold"}),
        ], style=PANEL_STYLE),

        html.Div([
            html.H4("4. Save Gate"),
            html.Label("Gate name:"),
            dcc.Input(id="gate-name", type="text",
                      placeholder="e.g. cd11c_gate or mhcii_cd11b_gate",
                      style={"width": "280px", "marginBottom": "8px",
                             "display": "block"}),
            html.Label("Save directory:"),
            dcc.Input(id="save-dir", type="text",
                      value="analysis/gates",  # EDIT FOR YOUR DATA: default save folder
                      style={"width": "280px", "marginBottom": "12px",
                             "display": "block"}),
            html.Button("Save Gate", id="save-btn",
                        style={"backgroundColor": "#d62728", "color": "white",
                               "border": "none", "padding": "8px 20px",
                               "borderRadius": "4px", "cursor": "pointer",
                               "marginBottom": "8px"}),
            html.Div(id="save-status",
                     style={"fontSize": "13px", "color": "#2ca02c",
                            "fontWeight": "bold", "marginTop": "8px"}),
            html.Hr(),
            html.H4("Saved Gates This Session"),
            html.Div(id="saved-gates-list",
                     style={"fontSize": "12px", "color": "#555",
                            "maxHeight": "120px", "overflowY": "auto"}),
        ], style=PANEL_STYLE),

    ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "flex-start"}),

    html.Hr(),

    # ── GATING MODE TOGGLE ───────────────────────────────────
    html.Div([
        html.Button(
            "🔴  GATING MODE: OFF — click to enable",
            id="gating-toggle-btn",
            n_clicks=0,
            style={
                "backgroundColor": "#888888",
                "color": "white",
                "border": "none",
                "padding": "10px 28px",
                "borderRadius": "6px",
                "cursor": "pointer",
                "fontSize": "15px",
                "fontWeight": "bold",
                "marginBottom": "8px",
                "marginRight": "12px",
            }
        ),
        html.Span(
            "When ON: clicks on the plot add gate points. "
            "When OFF: normal zoom/pan/hover.",
            style={"fontSize": "12px", "color": "#666", "verticalAlign": "middle"}
        ),
    ], style={"textAlign": "center", "marginBottom": "8px"}),

    # ── MAIN PLOT ────────────────────────────────────────────
    dcc.Graph(
        id="main-plot",
        style={"height": "600px"},
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": [],
        },
    ),

    # ── HIDDEN STORES ────────────────────────────────────────
    dcc.Store(id="poly-store",       data=[]),    # current polygon vertices
    dcc.Store(id="threshold-store",  data=None),  # current threshold value
    dcc.Store(id="saved-store",      data=[]),    # all saved gates this session
    dcc.Store(id="gating-active",    data=False), # gating mode on/off

], style={"fontFamily": "Arial, sans-serif", "maxWidth": "1600px",
          "margin": "0 auto", "padding": "20px"})


# ─────────────────────────────────────────────────────────────
# CALLBACK 0a — Gating mode toggle button
# ─────────────────────────────────────────────────────────────

@app.callback(
    Output("gating-active", "data"),
    Output("gating-toggle-btn", "children"),
    Output("gating-toggle-btn", "style"),
    Input("gating-toggle-btn", "n_clicks"),
    State("gating-active", "data"),
    prevent_initial_call=True,
)
def toggle_gating_mode(n_clicks, currently_active):
    now_active = not currently_active
    if now_active:
        label = "🟢  GATING MODE: ON — click plot to add points"
        style = {
            "backgroundColor": "#2ca02c",
            "color": "white",
            "border": "none",
            "padding": "10px 28px",
            "borderRadius": "6px",
            "cursor": "pointer",
            "fontSize": "15px",
            "fontWeight": "bold",
            "marginBottom": "8px",
            "marginRight": "12px",
        }
    else:
        label = "🔴  GATING MODE: OFF — click to enable"
        style = {
            "backgroundColor": "#888888",
            "color": "white",
            "border": "none",
            "padding": "10px 28px",
            "borderRadius": "6px",
            "cursor": "pointer",
            "fontSize": "15px",
            "fontWeight": "bold",
            "marginBottom": "8px",
            "marginRight": "12px",
        }
    return now_active, label, style


# ─────────────────────────────────────────────────────────────
# CALLBACK 0b — Switch dragmode based on gating active state
# ─────────────────────────────────────────────────────────────

@app.callback(
    Output("main-plot", "figure", allow_duplicate=True),
    Input("gating-active", "data"),
    State("main-plot", "figure"),
    prevent_initial_call=True,
)
def set_dragmode(gating_active, current_fig):
    if current_fig is None:
        return dash.no_update
    import plotly.graph_objects as go
    fig = go.Figure(current_fig)
    if gating_active:
        # Disable all drag interactions so clicks pass through as clickData
        fig.update_layout(dragmode=False)
    else:
        fig.update_layout(dragmode="zoom")
    return fig


# ─────────────────────────────────────────────────────────────
# CALLBACK 1 — Load FCS
# ─────────────────────────────────────────────────────────────

@app.callback(
    Output("x-channel", "options"),
    Output("x-channel", "value"),
    Output("y-channel", "options"),
    Output("y-channel", "value"),
    Output("load-status", "children"),
    Input("load-btn", "n_clicks"),
    State("fcs-path", "value"),
    prevent_initial_call=True,
)
def load_fcs(n_clicks, fcs_path):
    if not fcs_path or not os.path.exists(fcs_path):
        return [], None, [], None, f"❌ File not found: {fcs_path}"
    try:
        # If a directory is given, pool all FCS files in it directly
        if os.path.isdir(fcs_path):
            fcs_files = sorted(Path(fcs_path).glob("*.fcs"))
            if not fcs_files:
                return [], None, [], None, f"❌ No FCS files found in {fcs_path}"
            dfs = [read_fcs(str(f)) for f in fcs_files]
            df = pd.concat(dfs, ignore_index=True)
            CACHE["df_pooled"] = df
            CACHE["df_raw"]   = df
            CACHE["df_gated"] = df.copy()
            CACHE["fcs_path"] = fcs_path
            CACHE["channels"] = list(df.columns)
            # Build options and return early
            def is_useful(ch):
                upper = ch.upper()
                if any(upper.startswith(k) for k in ["FSC", "SSC", "TIME"]):
                    return True
                if upper.startswith("SS") and any(
                    f in upper for f in ["APC","BUV","BV","FITC","PE","PERCP","GHOST"]
                ):
                    return False
                import re
                if re.match(r"^(UV|V|B|R|YG|NIR)\d+-[AH]$", upper):
                    return False
                return True
            useful_cols = [c for c in df.columns if is_useful(c)]
            options = [{"label": c, "value": c} for c in useful_cols]
            def pick(keywords):
                for kw in keywords:
                    for c in useful_cols:
                        if kw.lower() in c.lower():
                            return c
                return useful_cols[0] if useful_cols else None
            x_default = pick(["FSC-A", "FSC"])
            y_default = pick(["SSC-A", "SSC"])
            msg = (f"✓ Pooled {len(fcs_files)} FCS files from {os.path.basename(fcs_path)}/ "
                   f"— {len(df):,} total events")
            return options, x_default, options, y_default, msg

        df = read_fcs(fcs_path)
        CACHE["df_raw"]   = df
        CACHE["df_gated"] = df.copy()
        CACHE["fcs_path"] = fcs_path
        CACHE["channels"] = list(df.columns)

        # Filter to useful channels only:
        # Exclude SS/bead spectral vectors (SS*, BEAD*) and raw detectors
        # (UV*, V*, B*, R*, YG*) — only show unmixed fluors + scatter
        def is_useful(ch):
            upper = ch.upper()
            # Always keep scatter
            if any(upper.startswith(k) for k in ["FSC", "SSC", "TIME"]):
                return True
            # Skip raw SS bead channels
            if upper.startswith("SS") and any(
                f in upper for f in ["APC","BUV","BV","FITC","PE","PERCP","GHOST"]
            ):
                return False
            # Skip raw detector channels (Cytek naming: UV1-A, V2-A, B3-A etc)
            import re
            if re.match(r"^(UV|V|B|R|YG|NIR)\d+-[AH]$", upper):
                return False
            return True

        useful_cols = [c for c in df.columns if is_useful(c)]
        options = [{"label": c, "value": c} for c in useful_cols]

        # Smart defaults: FSC/SSC if available, else first two channels
        def pick(keywords):
            for kw in keywords:
                for c in df.columns:
                    if kw.lower() in c.lower():
                        return c
            return df.columns[0]

        x_default = pick(["FSC-A", "FSC"])
        y_default = pick(["SSC-A", "SSC"])

        msg = (f"✓ Loaded {os.path.basename(fcs_path)} — "
               f"{len(df):,} events, {len(df.columns)} channels")

        # Auto-pool sibling wells if input is a folder of FCS files
        # (looks for other FCS files in the same directory with similar names)
        fcs_dir = os.path.dirname(fcs_path)
        siblings = sorted(Path(fcs_dir).glob("*.fcs"))
        if len(siblings) > 1:
            try:
                dfs = [read_fcs(str(s)) for s in siblings]
                df_pooled = pd.concat(dfs, ignore_index=True)
                CACHE["df_pooled"] = df_pooled
                n_pooled = len(df_pooled)
                msg += f" | {len(siblings)} files found in folder — pooled view available ({n_pooled:,} total events)"
            except Exception:
                CACHE["df_pooled"] = None
        else:
            CACHE["df_pooled"] = None

        return options, x_default, options, y_default, msg

    except Exception as e:
        return [], None, [], None, f"❌ Error: {e}"


# ─────────────────────────────────────────────────────────────
# CALLBACK 2 — Render plot + handle clicks
# ─────────────────────────────────────────────────────────────

@app.callback(
    Output("main-plot", "figure"),
    Output("poly-store", "data"),
    Output("threshold-store", "data"),
    Output("polygon-status", "children"),
    Input("main-plot", "clickData"),
    Input("x-channel", "value"),
    Input("y-channel", "value"),
    Input("transform", "value"),
    Input("plot-type", "value"),
    Input("subsample", "value"),
    Input("clear-btn", "n_clicks"),
    Input("undo-btn", "n_clicks"),
    Input("apply-gate-btn", "n_clicks"),
    State("gate-mode", "value"),
    State("threshold-direction", "value"),
    State("poly-store", "data"),
    State("threshold-store", "data"),
    State("fcs-overlay-path", "value"),
    State("gating-active", "data"),
    prevent_initial_call=True,
)
def update_plot(click_data, x_col, y_col, transform, plot_type,
                subsample, clear_clicks, undo_clicks, apply_clicks,
                gate_mode, threshold_direction,
                poly_pts, threshold_val, overlay_path, gating_active):
    CACHE["gating_active"] = gating_active

    triggered = ctx.triggered_id

    df = CACHE.get("df_gated")
    if df is None or x_col is None:
        return go.Figure(), poly_pts, threshold_val, "Load a file first."

    # ── Handle clear / undo ──
    if triggered == "clear-btn":
        poly_pts = []
        threshold_val = None
        CACHE["df_gated"] = CACHE["df_raw"].copy()

    if triggered == "undo-btn" and poly_pts:
        poly_pts = poly_pts[:-1]

    # ── Handle click → add point (only when gating mode is ON) ──
    new_threshold = threshold_val
    gating_on = CACHE.get("gating_active", False)
    if triggered == "main-plot" and click_data and gating_on:
        pt = click_data["points"][0]
        cx, cy = pt.get("x", 0), pt.get("y", 0)

        if gate_mode == "polygon":
            poly_pts = poly_pts + [(cx, cy)]
        elif gate_mode == "threshold":
            # In 1D mode x is the channel value
            new_threshold = cx

    # ── Apply gate to data ──
    if triggered == "apply-gate-btn":
        if gate_mode == "polygon" and len(poly_pts) >= 3:
            if y_col:
                x_raw = x_col
                y_raw = y_col
                # Work in raw space for gating (inverse transform if needed)
                if transform == "arcsinh":
                    # Store gate in transformed space — apply in transformed space
                    df_t = df.copy()
                    df_t[x_raw] = apply_transform(df[x_raw], transform)
                    df_t[y_raw] = apply_transform(df[y_raw], transform)
                    from shapely.geometry import Point, Polygon
                    poly = Polygon(poly_pts)
                    mask = [poly.contains(Point(x, y))
                            for x, y in zip(df_t[x_raw].values, df_t[y_raw].values)]
                    CACHE["df_gated"] = df[np.array(mask)]
                else:
                    CACHE["df_gated"] = apply_polygon_gate(df, poly_pts, x_raw, y_raw)
                df = CACHE["df_gated"]

        elif gate_mode == "threshold" and new_threshold is not None:
            if transform == "arcsinh":
                # threshold is in arcsinh space; compare arcsinh-transformed values
                vals_t = apply_transform(df[x_col], transform)
                if threshold_direction == "below":
                    mask = vals_t <= new_threshold
                else:
                    mask = vals_t >= new_threshold
                CACHE["df_gated"] = df[mask]
            else:
                CACHE["df_gated"] = apply_threshold_gate(
                    df, x_col, new_threshold, threshold_direction)
            df = CACHE["df_gated"]

    # ── Subsample for display ──
    df_plot = df if len(df) <= subsample else df.sample(subsample, random_state=42)

    # ── Transform values for display ──
    x_vals = apply_transform(df_plot[x_col], transform)
    x_label = f"{x_col} (arcsinh)" if transform == "arcsinh" else x_col

    # ── Build figure ──
    fig = go.Figure()

    is_1d = (y_col is None or y_col == x_col or gate_mode == "threshold")

    if is_1d:
        # Histogram
        fig.add_trace(go.Histogram(
            x=x_vals, nbinsx=200,
            name=f"Gated ({len(df):,} cells)",
            marker_color="#2166ac", opacity=0.7,
            histnorm="probability density",
        ))

        # Overlay FMO if provided
        if overlay_path and os.path.exists(overlay_path):
            try:
                df_ov = read_fcs(overlay_path)
                if x_col in df_ov.columns:
                    ov_vals = apply_transform(df_ov[x_col], transform)
                    fig.add_trace(go.Histogram(
                        x=ov_vals, nbinsx=200,
                        name="Overlay (FMO)",
                        marker_color="#d62728", opacity=0.5,
                        histnorm="probability density",
                    ))
            except Exception:
                pass

        # Invisible click-target for 1D threshold mode
        if gating_active:
            x_min = float(np.nanmin(x_vals)) if len(x_vals) else -1
            x_max = float(np.nanmax(x_vals)) if len(x_vals) else 8
            click_x = np.linspace(x_min, x_max, 200)
            fig.add_trace(go.Scatter(
                x=click_x,
                y=np.zeros(200),
                mode="markers",
                marker=dict(size=20, color="rgba(0,0,0,0)", opacity=0),
                hoverinfo="x",
                showlegend=False,
                name="__click_target__",
            ))

        # Threshold line
        if new_threshold is not None:
            fig.add_vline(x=new_threshold, line_dash="dash",
                          line_color="#e6550d", line_width=2,
                          annotation_text=f"Threshold: {new_threshold:.2f}",
                          annotation_position="top right")

        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title="Density",
            barmode="overlay",
        )

    else:
        # 2D scatter or density
        y_vals = apply_transform(df_plot[y_col], transform)
        y_label = f"{y_col} (arcsinh)" if transform == "arcsinh" else y_col

        if plot_type == "density":
            fig.add_trace(go.Histogram2dContour(
                x=x_vals, y=y_vals,
                colorscale="Viridis",
                reversescale=False,
                showscale=False,
                ncontours=20,
                name=f"Gated ({len(df):,} cells)",
            ))
            # Add scatter on top for click targets
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="markers",
                marker=dict(size=2, color="rgba(255,255,255,0.0)"),
                showlegend=False,
                hoverinfo="x+y",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="markers",
                marker=dict(size=2, color="#2166ac", opacity=0.3),
                name=f"Gated ({len(df):,} cells)",
            ))

        # Invisible dense click-target grid — covers whole plot area
        # so clickData fires on empty space, not just on data points
        if gating_active:
            x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
            y_min, y_max = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
            grid_x = np.linspace(x_min, x_max, 40)
            grid_y = np.linspace(y_min, y_max, 40)
            gx, gy = np.meshgrid(grid_x, grid_y)
            fig.add_trace(go.Scatter(
                x=gx.flatten(), y=gy.flatten(),
                mode="markers",
                marker=dict(size=18, color="rgba(0,0,0,0)", opacity=0),
                hoverinfo="x+y",
                showlegend=False,
                name="__click_target__",
            ))

        # Overlay FMO
        if overlay_path and os.path.exists(overlay_path):
            try:
                df_ov = read_fcs(overlay_path)
                if x_col in df_ov.columns and y_col in df_ov.columns:
                    ov_x = apply_transform(df_ov[x_col], transform)
                    ov_y = apply_transform(df_ov[y_col], transform)
                    fig.add_trace(go.Scatter(
                        x=ov_x, y=ov_y,
                        mode="markers",
                        marker=dict(size=2, color="#d62728", opacity=0.3),
                        name="Overlay (FMO)",
                    ))
            except Exception:
                pass

        # Draw polygon in progress
        if poly_pts and len(poly_pts) >= 1:
            px_list = [p[0] for p in poly_pts] + [poly_pts[0][0]]
            py_list = [p[1] for p in poly_pts] + [poly_pts[0][1]]
            fig.add_trace(go.Scatter(
                x=px_list, y=py_list,
                mode="lines+markers",
                line=dict(color="#e6550d", width=2, dash="dash"),
                marker=dict(size=8, color="#e6550d"),
                name="Gate polygon",
                showlegend=True,
            ))

        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label,
        )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=20, t=40, b=60),
        clickmode="event",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)

    # ── Status message ──
    n_raw   = len(CACHE["df_raw"]) if CACHE["df_raw"] is not None else 0
    n_gated = len(CACHE["df_gated"]) if CACHE["df_gated"] is not None else 0
    pct     = 100 * n_gated / n_raw if n_raw > 0 else 0

    if gate_mode == "polygon":
        status = (f"Polygon vertices: {len(poly_pts)}  "
                  f"(need ≥3 to gate)  |  "
                  f"Current population: {n_gated:,} / {n_raw:,} cells ({pct:.1f}%)")
    else:
        th_str = f"{new_threshold:.3f}" if new_threshold is not None else "none"
        status = (f"Threshold: {th_str}  |  "
                  f"Direction: {threshold_direction}  |  "
                  f"Current population: {n_gated:,} / {n_raw:,} cells ({pct:.1f}%)")

    return fig, poly_pts, new_threshold, status


# ─────────────────────────────────────────────────────────────
# CALLBACK 3 — Save gate
# ─────────────────────────────────────────────────────────────

@app.callback(
    Output("save-status", "children"),
    Output("saved-store", "data"),
    Output("saved-gates-list", "children"),
    Input("save-btn", "n_clicks"),
    State("gate-name", "value"),
    State("save-dir", "value"),
    State("gate-mode", "value"),
    State("poly-store", "data"),
    State("threshold-store", "data"),
    State("x-channel", "value"),
    State("threshold-direction", "value"),
    State("saved-store", "data"),
    prevent_initial_call=True,
)
def save_gate(n_clicks, gate_name, save_dir, gate_mode,
              poly_pts, threshold_val, x_col, direction, saved):

    if not gate_name:
        return "❌ Enter a gate name first.", saved, _render_saved(saved)

    os.makedirs(save_dir, exist_ok=True)
    saved = saved or []

    if gate_mode == "polygon":
        if len(poly_pts) < 3:
            return "❌ Need at least 3 polygon points.", saved, _render_saved(saved)
        out_path = os.path.join(save_dir, f"{gate_name}_hull.txt")
        save_polygon_txt(out_path, poly_pts)
        entry = {"type": "polygon", "name": gate_name, "path": out_path,
                 "n_points": len(poly_pts)}
        saved.append(entry)
        msg = f"✓ Saved polygon hull → {out_path}"

    else:  # threshold
        if threshold_val is None:
            return "❌ Click on histogram to set threshold first.", saved, _render_saved(saved)
        out_path = os.path.join(save_dir, f"{gate_name}_threshold.json")
        payload = {
            "gate_name":  gate_name,
            "channel":    x_col,
            "value":      threshold_val,
            "direction":  direction,
            "transform":  "arcsinh",
            "cofactor":   150,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        entry = {"type": "threshold", "name": gate_name, "path": out_path,
                 "channel": x_col, "value": threshold_val, "direction": direction}
        saved.append(entry)
        msg = f"✓ Saved threshold → {out_path}"

    return msg, saved, _render_saved(saved)


def _render_saved(saved):
    if not saved:
        return "None yet."
    items = []
    for s in saved:
        if s["type"] == "polygon":
            items.append(html.Div(f"[polygon] {s['name']} ({s['n_points']} pts) → {s['path']}"))
        else:
            items.append(html.Div(
                f"[threshold] {s['name']} | {s['channel']} "
                f"{s['direction']} {s['value']:.3f} → {s['path']}"
            ))
    return items


# ─────────────────────────────────────────────────────────────
# CALLBACK 4 — Gate stats readout
# ─────────────────────────────────────────────────────────────

@app.callback(
    Output("gate-stats", "children"),
    Input("apply-gate-btn", "n_clicks"),
    prevent_initial_call=True,
)
def update_stats(_):
    df_raw   = CACHE.get("df_raw")
    df_gated = CACHE.get("df_gated")
    if df_raw is None or df_gated is None:
        return ""
    n_raw   = len(df_raw)
    n_gated = len(df_gated)
    pct     = 100 * n_gated / n_raw if n_raw > 0 else 0
    return f"After gate: {n_gated:,} / {n_raw:,} cells ({pct:.1f}%)"


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nFlow View — Interactive Gating")
    print("Open http://127.0.0.1:8050 in your browser\n")
    app.run(debug=False, port=8050)
