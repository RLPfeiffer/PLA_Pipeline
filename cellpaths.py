"""
Path and configuration layer for multi-cell runs.

A project is a directory with a config file and three input directories. Cells
are discovered by matching filename patterns containing an {id} placeholder, so
you can point at whatever naming your exports already use rather than renaming
anything.

    project/
      cells.toml            paths, defaults, and per-cell overrides
      cells.csv             per-cell parameters, written on first discover
      <swc_dir>/            c192.swc, c476.swc, ...
      <mesh_dir>/           Morphology-192.dae, ...
      <synapse_dir>/        synapses-192.csv, ...
      out/
        192/
          regions.csv       review manifest - PRESERVED across re-runs
          repaired.swc
          branches.csv
          synapses.csv
          summary.csv
          repair.log
        all_branches.csv    aggregated across cells
        all_summary.csv
        run_manifest.csv    what ran, when, and whether it succeeded

Input directories may be anywhere, including read-only network shares. Nothing
is ever written outside the output directory.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

try:                        # stdlib from 3.11
    import tomllib
except ModuleNotFoundError:  # pip install tomli on 3.9 / 3.10
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "reading cells.toml needs tomllib (Python 3.11+) or the tomli "
            "backport. Run: pip install tomli") from None


CONFIG_NAME = "cells.toml"
PARAM_NAME = "cells.csv"

DEFAULT_CONFIG = r'''# swc pipeline configuration
# Paths may be absolute, or relative to this file's directory.
#
# WINDOWS PATHS: use SINGLE quotes. A single-quoted TOML string is literal, so
# backslashes are taken as-is and you can paste straight from Explorer:
#
#     swc_dir = 'E:\Data\Project\InputSkeletons'
#
# Double quotes will NOT work - TOML reads \D, \A etc. as escape sequences and
# the parse fails. Do not wrap the value in Python's r'...'; that is Python
# syntax, not TOML. Forward slashes in double quotes are also fine on Windows:
#
#     swc_dir = "E:/Data/Project/InputSkeletons"

# The volume this project covers. {volume} is substituted into every path and
# pattern below, so switching volumes is a one-line edit.
#
# Keep ONE PROJECT PER VOLUME. z_split, smooth_coef and section thickness are
# all volume-level properties, and cell ids can collide between volumes.
#
# Leave volume = "" to have {volume} in [patterns] act as a wildcard instead,
# captured from the filename and reported by `discover`. It cannot be a
# wildcard inside [paths] - a directory has to be named.
[project]
volume = "RPC1"

# Section thickness in micrometres. Used to convert the z_split_section column
# in cells.csv into micrometres, so you can enter the compartment boundary as
# the section number you read off in Viking rather than converting by hand.
#
# For this volume Z_um = section * 0.07 exactly, with no offset. Confirm for
# yours with `pipeline.py frames`, which prints the section range alongside the
# micrometre range.
section_thickness_um = 0.07

[paths]
swc_dir     = 'InputSkeletons/{volume}'
mesh_dir    = 'InputMeshes/{volume}'
synapse_dir = 'InputCSVs/{volume}'
out_dir     = "out"

# Filename patterns. {id} is the cell id and becomes a capture group when
# discovering, and a substitution when looking a known cell up. Give a list to
# accept more than one convention. Matching is case-insensitive.
[patterns]
swc     = ["c{id}.swc", "{id}.swc", "cell{id}.swc", "Cell-{id}.swc"]
mesh    = ["Morphology-{id}.dae", "c{id}.dae", "{id}.dae"]
# Note there is no bare "{id}.csv" here on purpose: it would match anything
# and hand you an id like 'c192-RPC1-linked-neurons'.
# No bare "{id}.csv" on purpose: it would match anything and hand you an id
# like 'c192-RPC1-linked-neurons' that never pairs with 'c192.swc'.
synapse = ["c{id}-{volume}-linked-neurons.csv", "synapses-{id}.csv",
           "c{id}_synapses.csv"]

# Defaults for every cell. Override per cell in cells.csv, which wins.
[defaults]
strip_types    = [6]      # SWC types that are appended markers, not neurite
soma_type      = 1
min_tree_dist  = 6.0      # duplicate detector: min separation along the tree
graft_within   = 2.0      # orphan fragments closer than this get grafted
drop_smaller_than = 1     # orphan fragments this size or smaller may be dropped
step           = 0.25     # resampling interval, um
# Coordinate rescaling for the SWC, applied as z_um = z_raw * z_scale + z_offset
# before anything else. Only needed if an SWC reports Z as a section INDEX
# rather than micrometres - run `pipeline.py frames` to find out. For a volume
# with 70 nm sections that would be z_scale = 0.07.
z_scale        = 1.0
z_offset       = 0.0
smooth_coef    = 0.8      # CALIBRATE THIS per volume against hand-traced branches
# Split the cell into lobular and arboreal compartments. ON, and it should stay
# on for density: the denominator has to be dendrite that could have received
# the input. Lobular dendrite never receives CBb input, so pooling the whole
# cell divides 48 arboreal CBb contacts by 603 um instead of 415, understating
# arboreal CBb density by 27%. The same argument applies to every partner class
# that is confined to one stratum.
use_compartments = true
z_split        = 0.0      # compartment boundary in Z, only used if the above
                          # is true. Enter it as z_split_section in cells.csv.
max_dist       = 3.0      # reject synapses farther than this from any centreline
group_by       = "type"   # type | partner | both
# Radius around the soma excluded from BOTH path length and synapse counts.
# -1 uses the soma node's own radius, which is the sensible default: collapsing
# the soma leaves radial spokes out to each primary dendrite, about one soma
# radius each, and they are not dendrite. 0 disables the exclusion.
exclude_soma_radius = -1.0

# Row filters for the synapse CSV. These are biological decisions and apply to
# the whole volume, so they live here rather than per cell.
#
# Run this first to see what your export actually contains:
#     python density.py <any-swc> <any-csv> --census
#
# A Viking linked-neurons export lists every annotated child structure, not
# just synapses. Desmosomes, multivesicular bodies, caveolae, cilia and
# endocytosis events are organelles and adhesions, and counting them as inputs
# roughly doubles the numbers. Direction=post restricts to input onto this cell;
# drop that line to keep the cell's own output as well.
# The detectors are advisory by default: audit reports what looks wrong and
# the Blender add-on draws it, but repair does not act on any of it. Fix the
# skeleton by hand in Blender, export edited.swc, and repair picks that up.
#
# Set apply_region_fixes = true if you would rather review regions.csv and let
# repair apply merge / graft / trim / snip for you.
[repair]
apply_region_fixes = false
prefer_edited = true

[filters]
include = { Direction = ["post"] }
# Types that bypass the include filters because they are bidirectional. A gap
# junction has no pre or post side: which one Viking's annotator marked is a
# convention. On cell 192 there are 110 distinct gap junctions split 77 post /
# 33 pre, so a blanket Direction=post filter would discard 30% of them.
bidirectional_types = ["GapJunction"]

exclude = { SynapseType = [
    "Desmosome", "MultivesicularBody", "Caveola", "Cilium",
    "Endocytosis", "NeuroglialAdherens", "Unknown",
] }

# Contact classes. A gap junction and a postsynaptic density are different kinds
# of contact and are reported separately in summary.csv: one is electrical
# coupling, the other chemical input. The same partner class often carries both
# - on cell 192, CBb has 40 gap junctions and 8 ribbons on the arboreal
# dendrites - so pooling them would be meaningless. Substring matching, first
# match wins, anything unmatched becomes "other".
[contact_classes]
gap_junction = ["GapJunction"]
psd          = ["RibbonPost", "ConvPost", "Post"]
presynaptic  = ["ConvPre", "CisternPre", "PlaqueLikePre", "Pre"]

# Depth plots. bin_um is the depth bin width for the profiles.
[plots]
bin_um = 2.0
# Partner classes with fewer than this many contacts are left off the depth
# panels. 1 shows all of them.
min_n = 1

# Which partner classes appear in each contact class's figure. Only the classes
# listed here are restricted; anything not mentioned shows every partner.
#
# PSDs are restricted to bipolar cells because that is what the analysis is
# about - amacrine input is the bulk of the PSDs and would flatten the BC
# series off the axis. Gap junctions are deliberately NOT restricted: the
# AII-AII and AII-CBb networks are both of interest. summary.csv keeps
# everything regardless.
[plots.labels]
psd = ["RodBC", "CBa", "CBb", "BC"]

# Partner label aliases, alias -> canonical. CBbwf is a wide-field CBb, not a
# separate partner class: counting it separately splits one population in two
# and leaves a one-contact series in every figure. Applied on load, so the
# summary, the grouping and the figures all agree.
[partner_aliases]
CBbwf = "CBb"

# Validation bands for the mesh cross-check, in Z, per compartment.
# Leave empty to skip. These are volume coordinates and are cell specific, so
# most projects will set them per cell in cells.csv instead.
[validation]
bands = []
'''

# Parameters that may be overridden per cell in cells.csv.
CELL_PARAMS = [
    "z_split_section", "z_split", "z_scale", "z_offset", "smooth_coef",
    "step", "max_dist", "group_by",
    "min_tree_dist", "graft_within", "drop_smaller_than",
    "exclude_soma_radius", "soma_type",
]
NUMERIC_PARAMS = {
    "z_split_section": float, "z_split": float, "z_scale": float, "z_offset": float, "smooth_coef": float, "step": float, "max_dist": float,
    "min_tree_dist": float, "graft_within": float, "drop_smaller_than": int,
    "exclude_soma_radius": float, "soma_type": int,
}


@dataclass
class Cell:
    cell_id: str
    volume: str = ""
    swc: Path | None = None
    mesh: Path | None = None
    synapses: Path | None = None
    params: dict = field(default_factory=dict)
    out_dir: Path | None = None

    # -- derived output paths ----------------------------------------
    @property
    def regions_csv(self):
        return self.out_dir / "regions.csv"

    @property
    def repaired_swc(self):
        return self.out_dir / "repaired.swc"

    @property
    def edited_swc(self):
        """Hand-fixed skeleton written by the Blender Skeleton Editor add-on."""
        return self.out_dir / "edited.swc"

    def repair_input(self, prefer_edited=True):
        """What repair should read: the hand-fixed skeleton if there is one."""
        if prefer_edited and self.edited_swc.exists():
            return self.edited_swc, "edited"
        return self.swc, "raw"

    @property
    def branches_csv(self):
        return self.out_dir / "branches.csv"

    @property
    def synapses_csv(self):
        return self.out_dir / "synapses.csv"

    @property
    def summary_csv(self):
        return self.out_dir / "summary.csv"

    @property
    def log(self):
        return self.out_dir / "repair.log"

    def density_prefix(self):
        return str(self.out_dir / "")[:-1] if False else str(self.out_dir)

    def missing(self):
        out = []
        if not self.swc:
            out.append("swc")
        if not self.synapses:
            out.append("synapses")
        return out

    def ready_for_density(self):
        return self.repaired_swc.exists() and self.synapses is not None

    def __str__(self):
        bits = [f"cell {self.cell_id}"
                + (f" [{self.volume}]" if self.volume else "")]
        for label, p in (("swc", self.swc), ("mesh", self.mesh),
                         ("syn", self.synapses)):
            bits.append(f"{label}={'-' if p is None else p.name}")
        return "  ".join(bits)


# Cell ids may not contain '-' or '.'. Without that restriction a permissive
# pattern like '{id}.csv' matches 'c192-RPC1-linked-neurons.csv' and reports the
# id as 'c192-RPC1-linked-neurons', which then fails to pair with 'c192.swc' -
# two half-populated cells instead of one, with no error anywhere.
ID_CHARS = "[A-Za-z0-9_]+?"


VOLUME_CHARS = "[A-Za-z0-9_]+?"


def _pattern_to_regex(pat, volume=None):
    """Turn 'c{id}-{volume}-linked.csv' into a case-insensitive regex.

    {id} is always a capture group. {volume} is substituted literally when the
    project declares a volume, and becomes a second capture group when it does
    not, so discovery still works on a mixed directory.
    """
    if volume:
        pat = pat.replace("{volume}", volume)
    chunks = re.split(r"(\{id\}|\{volume\})", pat)
    out = ["^"]
    for ch in chunks:
        if ch == "{id}":
            out.append(f"(?P<id>{ID_CHARS})")
        elif ch == "{volume}":
            out.append(f"(?P<volume>{VOLUME_CHARS})")
        else:
            out.append(re.escape(ch))
    out.append("$")
    return re.compile("".join(out), re.IGNORECASE)


class Project:
    def __init__(self, root, config=None):
        self.root = Path(root).expanduser().resolve()
        self.config_path = self.root / CONFIG_NAME
        self.cfg = config if config is not None else self._load_config()
        self.volume = str(self.cfg.get("project", {}).get("volume", "") or "")
        self.section_thickness = float(
            self.cfg.get("project", {}).get("section_thickness_um", 0.0) or 0.0)
        p = self.cfg.get("paths", {})
        self.swc_dir = self._resolve(p.get("swc_dir", "input/swc"))
        self.mesh_dir = self._resolve(p.get("mesh_dir", "input/mesh"))
        self.synapse_dir = self._resolve(p.get("synapse_dir", "input/synapses"))
        self.out_dir = self._resolve(p.get("out_dir", "out"))
        self.patterns = self.cfg.get("patterns", {})
        self.defaults = dict(self.cfg.get("defaults", {}))
        self.param_path = self.root / PARAM_NAME

    # -- setup -------------------------------------------------------

    @classmethod
    def init(cls, root, make_input_dirs=True):
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        cfg = root / CONFIG_NAME
        if not cfg.exists():
            cfg.write_text(DEFAULT_CONFIG)
        proj = cls(root)
        if make_input_dirs:
            for d in (proj.swc_dir, proj.mesh_dir, proj.synapse_dir):
                d.mkdir(parents=True, exist_ok=True)
        proj.out_dir.mkdir(parents=True, exist_ok=True)
        return proj

    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"no {CONFIG_NAME} in {self.root}. Run:  pipeline.py init {self.root}")
        with open(self.config_path, "rb") as fh:
            try:
                return tomllib.load(fh)
            except tomllib.TOMLDecodeError as e:
                hint = ""
                if "\\" in str(e) or "escape" in str(e).lower():
                    hint = ("\nA Windows path in double quotes is the usual "
                            "cause: TOML reads \\D, \\A etc. as escape "
                            "sequences. Use single quotes instead, which make "
                            "the string literal:\n"
                            "    swc_dir = 'E:\\Data\\Project\\InputSkeletons'")
                raise ValueError(f"{self.config_path}: {e}{hint}") from None

    @staticmethod
    def looks_absolute(value):
        """True for POSIX '/x', Windows 'E:\\x' or 'E:/x', and UNC '\\\\host\\share'.

        Path.is_absolute() answers for the host OS only, so on Linux or macOS a
        Windows path tests as relative and would be silently appended to the
        project root. That produced a nonsense path rather than an error, so
        absoluteness is decided by inspecting the string instead.
        """
        v = str(value)
        return bool(re.match(r"^([A-Za-z]:[\\/]|\\\\|/|~)", v))

    def _substitute(self, value):
        raw = str(value)
        if "{volume}" in raw:
            if not self.volume:
                raise ValueError(
                    f"{self.config_path}: path {raw!r} contains {{volume}} but "
                    f"[project] volume is empty. A directory cannot be a "
                    f"wildcard - set volume, or write the directory out.")
            raw = raw.replace("{volume}", self.volume)
        return raw

    def _resolve(self, value):
        raw = self._substitute(value)
        p = Path(raw).expanduser()
        if not self.looks_absolute(raw):
            return self.root / p
        if re.match(r"^([A-Za-z]:|\\\\)", raw) and os.name != "nt":
            raise ValueError(
                f"cells.toml has a Windows path ({raw!r}) but this is not "
                f"Windows. Either run the pipeline on Windows, or change the "
                f"path to this machine's mount point for that share.")
        return p

    # -- discovery ---------------------------------------------------

    def _scan(self, directory, kind):
        """{cell_id: Path} for every file in directory matching any pattern."""
        found = {}
        if not directory.is_dir():
            return found
        pats = self.patterns.get(kind, [])
        if isinstance(pats, str):
            pats = [pats]
        regexes = [_pattern_to_regex(p, self.volume) for p in pats]
        for entry in sorted(directory.iterdir()):
            if not entry.is_file():
                continue
            for rx in regexes:
                m = rx.match(entry.name)
                if m:
                    cid = m.group("id")
                    vol = (m.groupdict().get("volume") or self.volume or "")
                    found.setdefault(cid, (entry, vol))
                    break
        return found

    def discover(self, only=None):
        swc = self._scan(self.swc_dir, "swc")
        mesh = self._scan(self.mesh_dir, "mesh")
        syn = self._scan(self.synapse_dir, "synapse")
        overrides = self.read_params()

        def path_of(table, cid):
            hit = table.get(cid)
            return hit[0] if hit else None

        def vol_of(cid):
            for table in (syn, swc, mesh):
                hit = table.get(cid)
                if hit and hit[1]:
                    return hit[1]
            return self.volume

        ids = sorted(set(swc) | set(syn), key=_sort_key)
        if only:
            keep = {str(o) for o in only}
            ids = [i for i in ids if i in keep]
        cells = []
        for cid in ids:
            params = dict(self.defaults)
            params.update(overrides.get(cid, {}))
            params = self.resolve_z_split(cid, params)
            c = Cell(cell_id=cid, swc=path_of(swc, cid),
                     mesh=path_of(mesh, cid), synapses=path_of(syn, cid),
                     volume=vol_of(cid), params=params,
                     out_dir=self.out_dir / cid)
            cells.append(c)
        return cells

    def resolve_z_split(self, cell_id, params):
        """Turn z_split_section into z_split micrometres.

        A section number is what you actually read off in Viking, so that is
        the more natural thing to enter. If both columns are filled and they
        disagree, that is an error rather than a silent preference - one of the
        two is stale and guessing which would be worse than stopping.
        """
        sec = params.get("z_split_section")
        if sec in (None, "", 0) or float(sec) == 0.0:
            return params
        if not self.section_thickness:
            raise ValueError(
                f"cell {cell_id}: z_split_section is set but [project] "
                f"section_thickness_um is missing from {self.config_path}")
        derived = float(sec) * self.section_thickness
        existing = params.get("z_split")
        if existing not in (None, "", 0) and float(existing) != 0.0:
            if abs(float(existing) - derived) > 1e-6:
                raise ValueError(
                    f"cell {cell_id}: z_split_section {sec} implies "
                    f"z_split {derived:.4f} um, but z_split is set to "
                    f"{float(existing):.4f}. Clear one of the two columns.")
        params["z_split"] = derived
        params["_z_split_from_section"] = float(sec)
        return params

    def unmatched(self):
        """Files in the input directories that no pattern claimed - usually a
        naming convention the patterns do not cover yet."""
        out = {}
        for kind, d in (("swc", self.swc_dir), ("mesh", self.mesh_dir),
                        ("synapse", self.synapse_dir)):
            if not d.is_dir():
                out[kind] = []
                continue
            claimed = {v[0] for v in self._scan(d, kind).values()}
            out[kind] = [p.name for p in sorted(d.iterdir())
                         if p.is_file() and p not in claimed
                         and not p.name.startswith(".")]
        return out

    # -- per-cell parameter table ------------------------------------

    def read_params(self):
        if not self.param_path.exists():
            return {}
        out = {}
        with open(self.param_path, newline="") as fh:
            for row in csv.DictReader(fh):
                cid = (row.get("cell_id") or "").strip()
                if not cid:
                    continue
                vals = {}
                for k in CELL_PARAMS:
                    v = (row.get(k) or "").strip()
                    if v == "":
                        continue
                    conv = NUMERIC_PARAMS.get(k)
                    try:
                        vals[k] = conv(v) if conv else v
                    except ValueError:
                        pass
                out[cid] = vals
        return out

    def write_param_template(self, cells, overwrite=False):
        """Write cells.csv with one row per cell, blank means use the default.

        Never clobbers values you have already entered: existing rows are read
        back and merged.
        """
        existing = self.read_params()
        rows = []
        for c in cells:
            row = {"cell_id": c.cell_id}
            for k in CELL_PARAMS:
                row[k] = existing.get(c.cell_id, {}).get(k, "")
            row["notes"] = ""
            rows.append(row)
        if self.param_path.exists() and not overwrite:
            # preserve notes column too
            old_notes = {}
            with open(self.param_path, newline="") as fh:
                for r in csv.DictReader(fh):
                    old_notes[(r.get("cell_id") or "").strip()] = r.get("notes", "")
            for row in rows:
                row["notes"] = old_notes.get(row["cell_id"], "")
        with open(self.param_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["cell_id"] + CELL_PARAMS + ["notes"])
            w.writeheader()
            w.writerows(rows)
        return self.param_path

    # -- output ------------------------------------------------------

    def ensure_out(self, cell):
        cell.out_dir.mkdir(parents=True, exist_ok=True)
        return cell.out_dir

    def repair_options(self):
        r = self.cfg.get("repair", {})
        return dict(
            apply_region_fixes=bool(r.get("apply_region_fixes", False)),
            prefer_edited=bool(r.get("prefer_edited", True)),
        )

    def filters(self):
        """([(col, {vals})] include, [(col, {vals})] exclude) from [filters]."""
        f = self.cfg.get("filters", {})
        inc = [(c, set(v if isinstance(v, list) else [v]))
               for c, v in (f.get("include") or {}).items()]
        exc = [(c, set(v if isinstance(v, list) else [v]))
               for c, v in (f.get("exclude") or {}).items()]
        bid = f.get("bidirectional_types") or []
        if isinstance(bid, str):
            bid = [bid]
        return inc, exc, tuple(bid)

    def partner_aliases(self):
        """{alias: canonical} for folding partner labels, or None."""
        pa = self.cfg.get("partner_aliases")
        return {str(k): str(v) for k, v in pa.items()} if pa else None

    def contact_classes(self):
        """[(class name, [type substrings])] in match order, or None."""
        cc = self.cfg.get("contact_classes") or {}
        if not cc:
            return None
        return [(name, list(pats) if isinstance(pats, list) else [pats])
                for name, pats in cc.items()]

    @property
    def all_branches(self):
        return self.out_dir / "all_branches.csv"

    @property
    def all_summary(self):
        return self.out_dir / "all_summary.csv"

    @property
    def run_manifest(self):
        return self.out_dir / "run_manifest.csv"

    def describe(self):
        lines = [f"project root   {self.root}",
                 f"  config       {self.config_path}"
                 f"{'' if self.config_path.exists() else '   [MISSING]'}",
                 f"  volume       {self.volume or '(none, {volume} is a wildcard)'}",
                 f"  swc_dir      {self.swc_dir}"
                 f"{'' if self.swc_dir.is_dir() else '   [MISSING]'}",
                 f"  mesh_dir     {self.mesh_dir}"
                 f"{'' if self.mesh_dir.is_dir() else '   [MISSING]'}",
                 f"  synapse_dir  {self.synapse_dir}"
                 f"{'' if self.synapse_dir.is_dir() else '   [MISSING]'}",
                 f"  out_dir      {self.out_dir}"]
        return "\n".join(lines)


def _sort_key(cid):
    """Sort 192 before 1000, and numeric ids before alphabetic ones."""
    m = re.fullmatch(r"(\d+)", cid)
    return (0, int(m.group(1)), "") if m else (1, 0, cid)
