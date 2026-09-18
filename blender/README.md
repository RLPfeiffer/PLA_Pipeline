# Blender tools

Two scripts. One you install, one you paste in.

| file | what it is |
|---|---|
| `skeleton_editor.py` | **add-on.** Edit a skeleton by hand and write it back to SWC. |
| `blender_qc.py` | **script.** Read-only viewer: skeleton, mesh, synapses, density. |

Both need `skelgraph.py` and `cellpaths.py` from the repository root, one level
above this directory. Neither needs numpy or scipy, so nothing has to be
installed into Blender's Python.

---

## skeleton_editor.py

### Install

1. Keep this repository somewhere permanent.
2. Blender: **Edit → Preferences → Add-ons**.
3. Dropdown arrow at the top right (Blender 4.2+) or **Install...** (3.x), then
   **Install from Disk**.
4. Pick `skeleton_editor.py`.
5. Tick the checkbox next to **Skeleton Editor (SWC)**.

Then expand the add-on's entry and set:

- **Tools Directory** → the repository root, e.g. `E:\Code\PLA_Pipeline`.
  Not this `blender` folder — export imports `skelgraph.py` from the root.
- **Default Project** → your project directory. Optional, used by the
  *Fill from project* button.

Both settings live in Blender's preferences and persist across .blend files.

Blender **copies** the file into its own add-ons folder on install, so editing
the original afterwards has no effect. To update, reinstall over the top.

### Use

Press **N** in the 3D viewport, then the vertical **Skeleton** tab on the right
edge.

**Input** — three file pickers. Point **SWC** at whichever skeleton you want:
the raw export, an `edited.swc`, a `cleaned.swc`. Switching between versions to
compare is two clicks. **Mesh (.dae)** and **regions.csv** are optional
reference.

**Fill from project** asks for a cell ID and fills all four paths plus `z_split`
and the soma exclusion radius from `cells.toml`. It prefers `edited.swc`, then
`repaired.swc`, then the raw input.

**Output** — the **Export To** path is editable. **Beside input** sets it to
`<name>_edited.swc` next to the source, so you cannot overwrite your input by
accident. Missing directories are created.

**Import** does nothing if that object is already in the scene, so re-running is
safe. **Validate** checks the result is still a tree and writes nothing.
**Export** writes. **Re-import** reloads from disk and discards your edits,
after asking.

**Options** is collapsed by default: weld tolerance, true-calibre display,
reference mesh opacity, and the z-split plane and soma sphere.

### Editing

| key | what it does to the skeleton |
|---|---|
| `G` | move a node back onto the centreline |
| `X` → Vertices | delete a node, a stray, a whole twig |
| `E` | extrude, to extend a branch cut short |
| `F` | join two selected endpoints into one spline |
| right-click → Subdivide | add nodes along a segment |
| `Alt+S` | shrink/fatten — edits the radius, which is exported |
| `L` / `Ctrl+L` | select a whole spline; `L` then `X` deletes a branch |
| `X` → Segment | break a spline in two |

The skeleton is a **curve**, one poly spline per unbranched branch.

### Moving a branch point

A curve control point cannot have three neighbours, so a junction exists only as
**coincident endpoints**. You cannot create an edge; you move an endpoint onto
another.

1. Select the child branch's endpoint.
2. Turn on vertex snapping: `Shift+Tab`, then **Snap To → Vertex**.
3. `G` and drag it onto the parent control point you want.

Snapping matters. The weld tolerance is 0.02 µm by default, so eyeballing it
leaves the branch detached. **Validate** will tell you: an extra root in the
component list is a branch that did not reattach.

### What export guarantees

- **A cycle is refused**, naming a vertex so you can navigate to it. Every
  downstream tool assumes a tree and would loop forever on a cycle.
- **Stranded single points are dropped** — no neighbours means no path length.
- **Types are recovered by position**, so unmoved nodes keep theirs.
- **New points get a real radius.** Extrude and subdivide give Blender's default
  of exactly 1.0; where that value comes back on a point matching no original
  node, the radius is interpolated from its neighbours instead.
- **Node IDs are renumbered**, so re-run `pipeline.py audit` before reusing an
  old `regions.csv`.

### Display

`TRUE_CALIBRE` is off by default, giving a thin line with visible control
points. The radius is stored and exported either way, so this only changes what
you see. You can also set it live: Object Data Properties → Geometry → Bevel →
Depth. 0 for a bare centreline, 1.0 for true calibre.

`Alt+Z` for X-ray to see the skeleton inside the reference mesh.

---

## blender_qc.py

The read-only viewer. Use it to check that synapses landed on the branches you
expect, which is obvious on screen and invisible in a table.

Paste it into the Scripting workspace, set `TOOLS_DIR`, `PROJECT` and `CELL_ID`
in the CONFIG block at the top, and run it with `Alt+P`.

Use `r"..."` for the paths. These are Python strings, so a bare
`"E:\Code\..."` makes Python read `\C` as an escape sequence — it warns, and for
`\n`, `\t`, `\b`, `\f`, `\r`, `\v`, `\a`, `\x` or `\U` it silently corrupts the
path.

```python
TOOLS_DIR = r"E:\Code\PLA_Pipeline"
PROJECT   = r"E:\Data\Project"
CELL_ID   = "192"
```

It builds:

- `Mesh/` — the input `.dae`, unselectable
- `Skeleton/` — one curve per compartment, one spline per branch
- `Synapses/` — one object per category, spheres instanced on the points
- `Review/` — an empty per flagged region; select one and press Numpad-period
  to fly to it
- `Density/` — the skeleton re-split into colour-ramped quintiles, if you point
  it at `branches.csv`
- `MergePlan/` — only meaningful when `apply_region_fixes` is true

It computes nothing. Its branch decomposition is the same code path as
`density.py`, so what you see matches the tables.
