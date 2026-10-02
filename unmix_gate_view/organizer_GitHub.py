#organizer_GitHub.py — Hierarchy-driven organizer for spectral unmixing controls.
#!/usr/bin/env python3
"""
Hierarchy-driven organizer.py for spectral unmixing controls.

Decision hierarchy:
    0. Pooled controls (name starts with 'pooled')
    1. Filename tags (Unstained, FMO, (Cells), (Beads))
    2. Row identity (A/E/H = SS, B/F/H = FMO, C/D = Unstained)
    3. Column pairing (A1↔B1, E1↔F1, H1↔H1)
    4. Parent folder (cells/beads)

FMOs always inherit fluor name from their paired SS.
Pooled controls are always treated as unstained and never paired.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================
# 1. FILE DISCOVERY
# ============================================================

def discover_fcs_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Experiment folder not found: {root}")
    return sorted(root.rglob("*.fcs"))


# ============================================================
# 2. DATACLASSES
# ============================================================

@dataclass
class ControlFile:
    path: Path
    name: str
    well: Optional[str]
    row: Optional[str]
    col: Optional[str]
    group: str          # "beads" | "cells" | "unknown"
    type: str           # "ss" | "fmo" | "unstained" | "unknown"
    fluor: Optional[str]
    pooled: bool

    def __repr__(self) -> str:
        return (
            f"ControlFile({self.name!r}, well={self.well}, "
            f"group={self.group}, type={self.type}, fluor={self.fluor!r})"
        )


@dataclass
class ControlPair:
    ss: ControlFile
    fmo: ControlFile

    @property
    def fluor(self) -> str:
        return self.ss.fluor

    def __repr__(self) -> str:
        return f"ControlPair(fluor={self.fluor!r}, ss={self.ss.well}, fmo={self.fmo.well})"


@dataclass
class GroupControls:
    ss: list[ControlFile] = field(default_factory=list)
    fmo: list[ControlFile] = field(default_factory=list)
    unstained: Optional[ControlFile] = None
    pairs: list[ControlPair] = field(default_factory=list)
    missing_fmo: list[ControlFile] = field(default_factory=list)
    missing_ss: list[ControlFile] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not self.missing_fmo and not self.missing_ss


@dataclass
class ExperimentStructure:
    beads: GroupControls
    cells: GroupControls
    warnings: list[str] = field(default_factory=list)

    @property
    def all_pairs(self) -> list[ControlPair]:
        return self.beads.pairs + self.cells.pairs

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings) or not (self.beads.is_complete and self.cells.is_complete)

    def summary(self) -> str:
        lines = []
        for group_name in ("beads", "cells"):
            g: GroupControls = getattr(self, group_name)
            lines.append(
                f"{group_name}: {len(g.pairs)} pairs, "
                f"{len(g.missing_fmo)} missing FMO, "
                f"{len(g.missing_ss)} missing SS"
            )
        if self.warnings:
            lines.append("\nWarnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


# ============================================================
# 3. CLASSIFICATION HELPERS
# ============================================================

def clean_fluor_name(name: str) -> Optional[str]:
    """
    Extract fluor name from SS filenames.
    Handles patterns like:
        'APC (Beads)_Controls_gated.fcs'
        'PE-cy7_Controls_gated.fcs'
        'BV421_Controls'
        'Pooled_Ghost_Violet540_Cells_gated.fcs'
    """
    n = name

    # Remove extension
    n = re.sub(r"\.fcs$", "", n, flags=re.IGNORECASE)

    # Remove plate/gate suffixes
    n = re.sub(r"_Plate_.*$", "", n, flags=re.IGNORECASE)
    n = re.sub(r"_(gated?|hull|gate)$", "", n, flags=re.IGNORECASE)
    n = re.sub(r"[\s_]*(gate|gated|hull|controls?)$", "", n, flags=re.IGNORECASE)

    # Remove Pooled_ prefix
    n = re.sub(r"^Pooled_", "", n, flags=re.IGNORECASE)

    # Remove trailing _Cells / _Beads
    n = re.sub(r"[\s_]*(Cells?|Beads?)$", "", n, flags=re.IGNORECASE)

    # Remove leading well ID (A1, E7, etc.)
    n = re.sub(r"^[A-H]\d{1,2}\s*", "", n)

    # Remove (Cells) / (Beads)
    n = re.sub(r"\s*\((Cells?|Beads?)\)\s*", " ", n, flags=re.IGNORECASE)

    # Remove leftover 'Controls' anywhere
    n = re.sub(r"controls?", "", n, flags=re.IGNORECASE)

    # Remove leftover underscores/hyphens/spaces
    n = re.sub(r"[_\-\s]+", " ", n).strip()

    if not n:
        return None

    # Normalize case
    n = n.upper()

    # Fix common fluor formatting
    n = n.replace("PE CY7", "PE-CY7")
    n = n.replace("PE-CY7", "PE-CY7")
    n = n.replace("APC CY7", "APC-CY7")
    n = n.replace("APC-CY7", "APC-CY7")

    # Normalize Ghost Dye variants → canonical name
    if re.search(r"GHOST", n, re.IGNORECASE):
        n = "GHOST DYE VIOLET 540"

    return n



def classify_group(path: Path) -> str:
    """
    Determine beads vs cells vs unknown.

    Priority:
      0. Pooled name hint (pooled_*_beads vs pooled_*_cells)
      1. Filename tags ( (Cells), (Beads) )
      2. Parent folder name
    """
    name = path.name.lower()
    folder = path.parent.name.lower()

    # 0. Pooled hint
    if name.startswith("pooled"):
        if "bead" in name:
            return "beads"
        if "cell" in name:
            return "cells"

    # 1. Filename tags
    if "(beads" in name:
        return "beads"
    if "(cells" in name:
        return "cells"

    # 2. Folder name
    if "bead" in folder:
        return "beads"
    if "cell" in folder:
        return "cells"

    # 3. Row-based inference for FMO files with no explicit tag.
    # FMO files in row B typically pair with bead SS in row A.
    # Peek at sibling files in the same folder to confirm the group.
    if re.search(r"\bfmo\b", name, re.IGNORECASE):
        well_m = re.match(r"([A-H]\d{1,2})\b", path.name)
        if well_m:
            row = well_m.group(1)[0]
            fmo_to_ss_row = {"B": "A", "F": "E"}
            ss_row = fmo_to_ss_row.get(row)
            if ss_row:
                for sibling in path.parent.glob(f"{ss_row}*.fcs"):
                    sib = sibling.name.lower()
                    if "(beads" in sib:
                        return "beads"
                    if "(cells" in sib:
                        return "cells"

    return "unknown"


def classify_type(path: Path) -> str:
    """
    Hierarchical SS/FMO/UN classification:
        0. Pooled controls override everything
        1. Filename tags (Unstained, FMO)
        2. Row identity (A/E/H = SS, B/F/H = FMO, C/D = Unstained)
        3. Fallback = unknown
    """
    name = path.name.lower()
    well = extract_well_id(path.name)
    row = well[0] if well else None

    # 0. Pooled controls
    if name.startswith("pooled"):
        # Ghost Dye / viability pooled → SS (has a spectral signature)
        if any(k in name for k in ["ghost", "zombie", "live", "dead", "viab"]):
            return "ss"
        # Everything else pooled → unstained
        return "unstained"

    # 1. Filename tags
    if "unstain" in name:
        return "unstained"
    if re.search(r"\bfmo\b", name):
        return "fmo"

    # 1b. Full panel files → skip entirely (not a single color control)
    if "full" in name:
        return "unknown"

    # 1c. Ghost Dye / viability keyword → SS regardless of row
    if any(k in name for k in ["ghost", "zombie", "live", "dead", "viab"]):
        return "ss"

    # 2. Row identity
    if row in ["C", "D"]:
        return "unstained"
    if row in ["A", "E", "H"]:
        return "ss"
    if row in ["B", "F", "H"]:
        return "fmo"

    # 3. Fallback
    return "unknown"

def extract_well_id(name: str) -> Optional[str]:
    m = re.match(r"([A-H]\d{1,2})\b", name)
    return m.group(1) if m else None


def extract_metadata(path: Path) -> ControlFile:
    name = path.name
    well = extract_well_id(name)
    row = well[0] if well else None
    col = well[1:] if well else None

    group = classify_group(path)
    ctype = classify_type(path)

    # Only SS files get a fluor from filename; FMOs inherit later; unstained has None
    fluor = clean_fluor_name(name) if ctype == "ss" else None

    return ControlFile(
        path=path,
        name=name,
        well=well,
        row=row,
        col=col,
        group=group,
        type=ctype,
        fluor=fluor,
        pooled=name.lower().startswith("pooled"),
    )


# ============================================================
# 4. PAIRING LOGIC
# ============================================================

# Row pairs for SS↔FMO
ROW_PAIRS = [
    ("A", "B"),
    ("E", "F"),
    ("H", "H"),
]


def pair_controls(
    ss_list: list[ControlFile],
    fmo_list: list[ControlFile],
    warnings: list[str],
) -> tuple[list[ControlPair], list[ControlFile], list[ControlFile]]:

    # Ignore pooled/unstained here; they have type "unstained" and won't be in ss_list/fmo_list
    ss_by_well = {m.well: m for m in ss_list if m.well}
    fmo_by_well = {m.well: m for m in fmo_list if m.well}

    pairs: list[ControlPair] = []
    missing_fmo: list[ControlFile] = []
    missing_ss: list[ControlFile] = []

    # SS → FMO by row pair + column
    for ss_row, fmo_row in ROW_PAIRS:
        for ss_meta in (m for m in ss_list if m.row == ss_row):
            fmo_well = f"{fmo_row}{ss_meta.col}"

            if fmo_well in fmo_by_well:
                # FMO inherits fluor from SS
                fmo_by_well[fmo_well].fluor = ss_meta.fluor
                pairs.append(ControlPair(ss=ss_meta, fmo=fmo_by_well[fmo_well]))
            else:
                missing_fmo.append(ss_meta)

    # Pooled SS files (no well ID) have no FMO — add directly to missing_fmo
    for ss_meta in ss_list:
        if ss_meta.well is None and ss_meta.fluor:
            missing_fmo.append(ss_meta)

    # Orphan FMOs (no matching SS)
    paired_fmo_wells = {p.fmo.well for p in pairs}
    for fmo_meta in fmo_list:
        if fmo_meta.well not in paired_fmo_wells:
            missing_ss.append(fmo_meta)

    return pairs, missing_fmo, missing_ss


# ============================================================
# 5. FINAL STRUCTURE ASSEMBLY
# ============================================================

def assemble_structure(
    metadata: list[ControlFile],
    warnings: list[str],
) -> ExperimentStructure:

    def _get(group: str, ftype: str) -> list[ControlFile]:
        return [m for m in metadata if m.group == group and m.type == ftype]

    def _unstained(group: str) -> Optional[ControlFile]:
        files = _get(group, "unstained")
        if len(files) > 1:
            warnings.append(
                f"Multiple unstained files for {group!r}; using {files[0].name!r}. "
                f"Others ignored: {[f.name for f in files[1:]]}"
            )
        return files[0] if files else None

    bead_pairs, bead_missing_fmo, bead_missing_ss = pair_controls(
        _get("beads", "ss"), _get("beads", "fmo"), warnings
    )
    cell_pairs, cell_missing_fmo, cell_missing_ss = pair_controls(
        _get("cells", "ss"), _get("cells", "fmo"), warnings
    )

    return ExperimentStructure(
        beads=GroupControls(
            ss=_get("beads", "ss"),
            fmo=_get("beads", "fmo"),
            unstained=_unstained("beads"),
            pairs=bead_pairs,
            missing_fmo=bead_missing_fmo,
            missing_ss=bead_missing_ss,
        ),
        cells=GroupControls(
            ss=_get("cells", "ss"),
            fmo=_get("cells", "fmo"),
            unstained=_unstained("cells"),
            pairs=cell_pairs,
            missing_fmo=cell_missing_fmo,
            missing_ss=cell_missing_ss,
        ),
        warnings=warnings,
    )


# ============================================================
# 6. MAIN
# ============================================================

def organize(experiment_root: Path) -> ExperimentStructure:
    warnings: list[str] = []
    files = discover_fcs_files(experiment_root)
    metadata = [extract_metadata(f) for f in files]
    return assemble_structure(metadata, warnings)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Organize gated controls for spectral unmixing.")
    parser.add_argument("experiment_root")
    args = parser.parse_args()

    structure = organize(Path(args.experiment_root))
    print(structure.summary())
