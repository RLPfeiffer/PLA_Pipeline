# PLA Pipeline

Measures synapse density per micrometre of dendritic path length, for neurons
reconstructed in Viking and exported with SBFSEM-pytools.

Takes three files per cell — an SWC skeleton, a COLLADA mesh and a
linked-neurons CSV — repairs the skeleton, assigns each synapse to a branch, and
reports density by compartment, contact class and partner cell class.

Built for AII amacrine cells in RPC1, but nothing in it is specific to that.

---

## Install

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9+. Needs numpy, scipy and matplotlib.

Use `requirements-lock.txt` instead for a reproducible run: it pins the exact
versions the pipeline was verified against.

---

## Quick start

```
python pipeline.py init     /path/to/project
python pipeline.py paths    /path/to/project
python pipeline.py discover /path/to/project
```

Edit `/path/to/project/cells.csv` and set `z_split_section` for each cell, then:

```
python pipeline.py run /path/to/project
```

Results land in `/path/to/project/out/`.

---

## How it is organised

A **project** is a directory holding a config file, a per-cell parameter table,
and the outputs. Your exports stay wherever they already are; the config points
at them.

```
project/
  cells.toml          paths, filters, contact classes
  cells.csv           per-cell parameters
  out/
    192/              one folder per cell
      regions.csv       what the detectors flagged
      repaired.swc      the analysis-ready skeleton
      edited.swc        your hand-fixed skeleton, if you made one
      branches.csv      per-branch length and counts
      synapses.csv      per-synapse assignment
      summary.csv       the numbers
      depth_*.png       depth profiles
      repair.log        full transcript
    all_summary.csv     every cell, concatenated
    all_branches.csv
    summary_RPC1.png    project-level bars, one figure per volume
    run_manifest.csv    what ran and whether it worked
```

Keep **one project per volume**. Section thickness and the smoothing
calibration are volume-level properties, and cell IDs can collide between
volumes.

---

## Setup in detail

### 1. Scaffold

```
python pipeline.py init /path/to/project
```

Writes `cells.toml`. Edit its `[paths]` section to point at your export
directories.

On Windows use **single quotes** — a single-quoted TOML string is literal, so
backslashes pass through untouched:

```toml
[project]
volume = "RPC1"
section_thickness_um = 0.07

[paths]
swc_dir     = 'E:\Data\Project\InputSkeletons\{volume}'
mesh_dir    = 'E:\Data\Project\InputMeshes\{volume}'
synapse_dir = 'E:\Data\Project\InputCSVs\{volume}'
out_dir     = "out"
```

Double quotes fail: TOML reads `\D` as an escape sequence. Forward slashes in
double quotes work too.

`{volume}` is substituted into every path and pattern, so switching volumes is
a one-line edit.

### 2. Check the paths

```
python pipeline.py paths /path/to/project
```

Confirms the config file it read, the resolved directories, and how many files
matched in each. It also lists files that no pattern claimed — that is how you
find out a naming convention is missing.

Filename patterns live in `[patterns]` and use `{id}` for the cell ID:

```toml
[patterns]
swc     = ["c{id}.swc"]
mesh    = ["Morphology-{id}.dae"]
synapse = ["c{id}-{volume}-linked-neurons.csv"]
```

### 3. Check the coordinate frames

```
python pipeline.py frames /path/to/project
```

Run this on every new export. It prints the X, Y and Z range of all three files
per cell, plus the section range and whether the section thickness divides
cleanly.

If the three sources disagree, no synapse lands near any centreline, every row
goes to UNASSIGNED, and the summary comes out full of zeros rather than
erroring. Fix a mismatch with `z_scale` and `z_offset` in `cells.csv`.

### 4. See what is in the synapse CSV

```
python pipeline.py census /path/to/project
```

Lists every column and, for the categorical ones, its value counts.

A linked-neurons export lists **every annotated child structure**, not just
synapses — desmosomes, multivesicular bodies, caveolae and so on. The default
`[filters]` block drops those; check the census against it and add anything
your volume has that the defaults miss.

### 5. Fill in the parameters

```
python pipeline.py discover /path/to/project
```

Writes `cells.csv`, one row per cell. Blank means "use the default from
`cells.toml`".

Set two things per cell:

| column | what it is |
|---|---|
| `z_split_section` | the section number at the boundary between the two dendritic strata |
| `group_by` | which CSV column to break the results down by, usually `NeuronLabel` |

Re-run `discover` and check each line echoes the conversion:

```
cell 192 [RPC1]  swc=c192.swc  mesh=Morphology-192.dae  ...  z_split section 839 = 58.730 um
```

`discover` is safe to re-run; it preserves values you have already entered.

---

## Running it

```
python pipeline.py run /path/to/project
```

Does audit, repair, density, plots and aggregate for every cell. Add
`--cells 192 476` to restrict it. Each stage can also be run on its own:

| command | what it does |
|---|---|
| `audit` | report skeleton problems, write `regions.csv` |
| `repair` | fix the skeleton, write `repaired.swc` |
| `density` | assign synapses, write the tables |
| `plots` | depth profiles per cell |
| `aggregate` | concatenate cells, draw the project bars |
| `summary` | just redraw the project bars |

---

## Fixing a skeleton by hand

The detectors are heuristics and are **advisory by default** — `audit` reports
what looks wrong, but `repair` acts on none of it.

Install the Blender add-on (see `blender/README.md`), load the cell, fix it, and
export. `repair` picks up `out/<id>/edited.swc` automatically in preference to
the raw input, and says so in the log.

```
audit  ->  edit in Blender  ->  repair  ->  density
```

Set `apply_region_fixes = true` in `cells.toml` if you would rather review
`regions.csv` and have `repair` apply the fixes for you.

---

## What repair does

In this order, because geometry fixes are meaningless before topology fixes:

1. strip appended marker nodes (SWC type 6 by default)
2. collapse the soma node chain to one node
3. apply detector fixes, if enabled
4. graft or drop disconnected fragments
5. split multifurcations into nested bifurcations
6. smooth and resample at fixed arc length

Every operation is validated and rolled back if it would leave an invalid tree.

The skeletons come from per-section contour centroids, and the centroid wanders
laterally by about as much as the section spacing advances axially. Smoothing
corrects that. The strength is set by `smooth_coef`, which **must be calibrated
per volume** against a handful of hand-traced branches:

```python
import swcrepair as sr
sk = sr.Skel.load('out/192/repaired.swc')
best, table = sr.calibrate_smoothing(sk, {79: 4.2, 451: 3.9, 557: 6.8})
```

Keys are tip node IDs, values your measured lengths in µm. Until this is done,
densities are self-consistent across cells — so relative comparisons hold — but
absolute values carry an unknown scale factor.

---

## Reading the results

`summary.csv` has one row per compartment × contact class × category:

| column | meaning |
|---|---|
| `volume`, `cell_id` | which cell |
| `compartment` | `lobular` or `arboreal`, from each branch's mean Z |
| `contact_class` | `gap_junction`, `psd`, `presynaptic` or `other` |
| `category` | whatever `group_by` selected, usually the partner cell class |
| `n` | synapses in that combination |
| `n_branches` | branches in that compartment |
| `length_um` | dendritic path length in that compartment |
| `density_per_um` | `n / length_um` |

`n_branches` and `length_um` are compartment totals, repeated on every row of
that compartment — they are the denominator, not per-category quantities.

### Three checks before trusting a cell

1. **Unassigned synapses under 5%.** The run warns above that. A high figure
   means `max_dist` is too small, the skeleton misses the distal tips, or the
   two files are in different coordinate frames.
2. **`residual_um` near the local process radius**, in `synapses.csv`. That is
   the distance from each synapse to the centreline it was assigned to.
3. **The stratification looks right.** For an AII, rod bipolar input should be
   arboreal and OFF cone bipolar contacts lobular. If those smear across both
   compartments, suspect `z_split_section` before suspecting the biology.

---

## Configuration reference

All in `cells.toml`.

```toml
[project]
volume = "RPC1"
section_thickness_um = 0.07     # converts z_split_section to micrometres

[filters]
include = { Direction = ["post"] }
exclude = { SynapseType = ["Desmosome", "MultivesicularBody", ...] }
bidirectional_types = ["GapJunction"]

[contact_classes]
gap_junction = ["GapJunction"]
psd          = ["RibbonPost", "ConvPost", "Post"]
presynaptic  = ["ConvPre", "CisternPre", "PlaqueLikePre", "Pre"]

[partner_aliases]
CBbwf = "CBb"

[plots]
bin_um = 2.0
min_n  = 1

[plots.labels]
psd = ["RodBC", "CBa", "CBb", "BC"]
```

**`bidirectional_types`** matters. A gap junction has no pre or post side, so
which one was annotated is a convention — rows matching these types bypass the
`include` filters. Without it, a `Direction = post` filter silently discards
roughly a third of the gap junctions.

**`[contact_classes]`** keeps gap junctions and postsynaptic densities apart in
the summary and in separate figures. The same partner class routinely carries
both, and they are different kinds of contact.

**`[plots.labels]`** restricts which partner classes appear in a figure. PSDs
are limited to bipolar cells by default because amacrine input is the bulk of
them and flattens the rest off the axis. Gap junctions are unrestricted.
`summary.csv` keeps everything regardless.

Per-cell overrides go in `cells.csv`: `z_split_section`, `z_split`, `z_scale`,
`z_offset`, `smooth_coef`, `step`, `max_dist`, `group_by`, `min_tree_dist`,
`graft_within`, `drop_smaller_than`, `exclude_soma_radius`, `soma_type`.

---

## Files

| file | what it is |
|---|---|
| `pipeline.py` | the driver — run everything through this |
| `cellpaths.py` | config and file discovery |
| `swcrepair.py` | detectors and repair operations |
| `repair_run.py` | the audit and repair stages |
| `density.py` | synapse assignment and the density tables |
| `plots.py` | depth profiles and project bars |
| `skelgraph.py` | SWC tree rebuilding, used by the Blender add-on |
| `blender/` | the Blender add-on and viewer |
