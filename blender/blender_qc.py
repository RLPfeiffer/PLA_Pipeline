"""
Blender QC viewer for the repaired skeleton, the synapses, and the review regions.

This does NOT compute anything. Counting and density happen in density.py, where
they are fast, testable and reproducible. Blender's job is the three things it is
actually better at than a script:

  1. flying to each row of regions.csv to decide merge vs keep
  2. checking that synapses landed on the branch you think they did
  3. tracing the 3-5 reference branches that calibrate the smoothing

HOW TO USE
  Scripting workspace -> New -> paste this -> set the paths in CONFIG -> Run Script
  (Alt+P). Nothing here needs the files to be in the blend's folder.

WHAT YOU GET
  Mesh/               the input .dae, so you can see the skeleton inside it
  Skeleton/           one curve object per compartment, one spline per branch
  Synapses/           one object per type, spheres instanced on synapse points
  Review/             an empty at each flagged region, named by id
  MergePlan/          only meaningful when apply_region_fixes is true in
                      cells.toml. Per region, colour coded by what would
                      change:
                        KEEP_<id>     green    survives a merge
                        DROP_<id>     red      deleted by a merge
                        FRAGMENT_<id> cyan     stranded, graft joins it on
                        HAIRPIN_<id>  orange   a fold, trim deletes the interior
                        OFFMESH_<id>  magenta  outside the membrane, snip deletes
                        SPIKE_<id>    red      excursion off course, snip deletes
                      Off-mesh and spike nodes get spheres so a single bad
                      node is findable; the thin curve either side is context.
  Density/            optional: skeleton re-split into density quintiles

  Select an empty in Review/ and press Numpad-period to fly to it.
  In the N-panel Item tab, an empty's name carries its region id, so you can
  find the matching row in regions.csv.
"""

import csv
import os
from pathlib import Path

import bpy
from mathutils import Vector

# ----------------------------------------------------------------- CONFIG
#
# Two ways to point this at files. Either set PROJECT + CELL_ID and let it
# resolve everything from cells.toml (preferred, and it picks up the per-cell
# z_split so the compartment split matches the tables exactly), or leave
# PROJECT empty and fill in the four explicit paths below.

# WINDOWS PATHS: use an r"..." prefix. These are Python strings, so a bare
# "E:\Code\..." makes Python read \C as an escape sequence - it warns, and for
# \n, \t, \b, \f, \r, \v, \a, \x or \U it silently corrupts the path.
# Forward slashes work too.
TOOLS_DIR    = r"E:\Code\PLA_Pipeline"
PROJECT      = r"E:\Documents\Data\RPC1\Aii_GAC_Project\PathLengthAnalysis_2026"
CELL_ID      = "192"     # change this per cell; everything else follows

SWC          = r""       # used when PROJECT is empty
MESH_DAE     = r""       # or "" to skip
SYNAPSE_CSV  = r""       # or "" to skip
REGIONS_CSV  = r""       # or "" to skip
BRANCHES_CSV = r""       # or "" to skip density colouring

Z_SPLIT      = 59.5      # overridden by the project's per-cell value if PROJECT set

LOAD_MESH    = True      # bring in the .dae so you can see the skeleton inside it
MESH_ALPHA   = 0.15      # mesh transparency. Press Alt+Z for X-ray if it looks solid
MESH_DECIMATE = 0.0      # 0 = full mesh. Set e.g. 0.25 if 300k triangles is sluggish
SKEL_RADIUS  = 0.06      # curve bevel depth, um. purely cosmetic
SYN_RADIUS   = 0.25      # synapse sphere radius, um
RECENTRE     = True      # move everything so the soma sits at the world origin

TYPE_COLOURS = {         # matched case-insensitively against the type column
    "ribbon":       (0.16, 0.47, 0.84, 1.0),
    "conventional": (0.92, 0.41, 0.20, 1.0),
    "gap junction": (0.11, 0.69, 0.48, 1.0),
    "gap":          (0.11, 0.69, 0.48, 1.0),
}
FALLBACK_COLOURS = [
    (0.93, 0.63, 0.00, 1.0), (0.91, 0.48, 0.64, 1.0),
    (0.38, 0.31, 0.84, 1.0), (0.89, 0.29, 0.29, 1.0),
    (0.00, 0.51, 0.00, 1.0),
]


# ------------------------------------------------------------------ helpers

def collection(name, parent=None):
    if name in bpy.data.collections:
        c = bpy.data.collections[name]
    else:
        c = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(c)
    return c


def material(name, rgba):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    m = bpy.data.materials.new(name)
    try:                        # removed in Blender 6.0, nodes are the default
        if not getattr(m, "use_nodes", True):
            m.use_nodes = True
    except (AttributeError, TypeError):
        pass
    tree = getattr(m, "node_tree", None)
    bsdf = tree.nodes.get("Principled BSDF") if tree else None
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.5
    m.diffuse_color = rgba          # so Solid shading shows it too
    return m


def load_swc(path):
    pos, rad, typ, par = {}, {}, {}, {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split()
            if len(f) < 7:
                continue
            i = int(f[0])
            typ[i] = int(f[1])
            pos[i] = Vector((float(f[2]), float(f[3]), float(f[4])))
            rad[i] = float(f[5])
            par[i] = int(f[6])
    return pos, rad, typ, par


def unbranched_paths(pos, par):
    """One list per branch: root/branch-point through to branch-point or tip."""
    ch = {}
    for i, p in par.items():
        if p != -1:
            ch.setdefault(p, []).append(i)
    starts = [i for i in pos if par[i] == -1 or len(ch.get(i, [])) >= 2]
    out = []
    for s in starts:
        for c in ch.get(s, []):
            path = [s, c]
            while len(ch.get(path[-1], [])) == 1:
                path.append(ch[path[-1]][0])
            out.append(path)
    return out


def make_curve(name, polylines, coll, mat, offset, radius):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 2
    for pl in polylines:
        if len(pl) < 2:
            continue
        sp = cu.splines.new("POLY")
        sp.points.add(len(pl) - 1)
        for k, v in enumerate(pl):
            q = v - offset
            sp.points[k].co = (q.x, q.y, q.z, 1.0)
    ob = bpy.data.objects.new(name, cu)
    ob.data.materials.append(mat)
    coll.objects.link(ob)
    return ob


def make_points(name, points, coll, mat, offset, radius):
    """A vertex cloud with a sphere instanced on each vertex - fast and clickable."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p - offset) for p in points], [], [])
    me.update()
    holder = bpy.data.objects.new(name, me)
    holder.instance_type = "VERTS"
    coll.objects.link(holder)

    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius,
                                          location=(0, 0, 0))
    sph = bpy.context.active_object
    sph.name = name + "_marker"
    for c in list(sph.users_collection):
        c.objects.unlink(sph)
    coll.objects.link(sph)
    sph.data.materials.clear()
    sph.data.materials.append(mat)
    sph.parent = holder
    return holder


def load_dae(path, to_volume_coords=True):
    """Parse a Viking COLLADA export into (verts, tris).

    Deliberately not bpy.ops.wm.collada_import: that applies the file's <unit>
    tag, which in Viking exports is the source volume's pixel resolution rather
    than a real scale factor, and it would land the mesh in a different frame
    from the skeleton. This mirrors swcrepair.load_dae_mesh exactly, so the two
    are guaranteed to be in the same coordinates.

    Takes about half a second on a 300k-triangle export.
    """
    import re as _re
    data = open(path, "r", errors="replace").read()
    pm = _re.search(r'positions-array" count="(\d+)">(.*?)</float_array>',
                    data, _re.S)
    tm = _re.search(r'<triangles count="(\d+)".*?<p>(.*?)</p>', data, _re.S)
    if not (pm and tm):
        raise RuntimeError(f"could not find geometry in {path}")
    flat = [float(v) for v in pm.group(2).split()]
    verts = [tuple(flat[i:i + 3]) for i in range(0, len(flat), 3)]
    idx = [int(v) for v in tm.group(2).split()]
    tris = [tuple(idx[i:i + 3]) for i in range(0, len(idx), 3)]
    if to_volume_coords:
        nm = _re.search(r"<translate>([^<]+)</translate>", data)
        if nm:
            ox, oy, oz = (float(v) for v in nm.group(1).split())
            verts = [(x + ox, y + oy, z + oz) for x, y, z in verts]
    return verts, tris


def colour_for(label, seen):
    key = label.strip().lower()
    for k, v in TYPE_COLOURS.items():
        if k in key:
            return v
    if key not in seen:
        seen[key] = FALLBACK_COLOURS[len(seen) % len(FALLBACK_COLOURS)]
    return seen[key]


# Kept in sync with density.CANDIDATES by hand. Blender does not ship scipy,
# so density.py cannot be imported here even when TOOLS_DIR is set.
CANDIDATES = {
    "x": ["x", "xum", "volumex", "xvolume", "posx", "centroidx", "synapsex",
          "synapsexum", "locationx"],
    "y": ["y", "yum", "volumey", "yvolume", "posy", "centroidy", "synapsey",
          "synapseyum", "locationy"],
    "z": ["z", "zum", "volumez", "zvolume", "posz", "centroidz", "synapsez",
          "synapsezum", "locationz", "section"],
    "type": ["synapsetype", "structuretype", "type", "label", "tag", "name"],
}
COORD_SUFFIXES = {"x": ("x", "xum"), "y": ("y", "yum"), "z": ("z", "zum")}


def sniff(fieldnames, want):
    """Find the column for 'x', 'y', 'z' or 'type'. Mirrors density.py."""
    table = {"".join(ch for ch in f.lower() if ch.isalnum()): f
             for f in fieldnames}
    for cand in CANDIDATES.get(want, [want]):
        if cand in table:
            return table[cand]
    for suffix in COORD_SUFFIXES.get(want, ()):
        for k, f in table.items():
            if k.endswith(suffix):
                return f
    return None


# --------------------------------------------------------------------- run

def resolve_paths():
    """Return (swc, synapse_csv, regions_csv, branches_csv, z_split)."""
    if not PROJECT:
        return (SWC, SYNAPSE_CSV, REGIONS_CSV, BRANCHES_CSV, Z_SPLIT, MESH_DAE)
    import sys
    if TOOLS_DIR and TOOLS_DIR not in sys.path:
        sys.path.insert(0, TOOLS_DIR)
    try:
        import cellpaths
    except ImportError:
        raise RuntimeError(
            "PROJECT is set but cellpaths.py could not be imported. "
            "Set TOOLS_DIR to the directory containing it.")
    proj = cellpaths.Project(PROJECT)
    cells = {c.cell_id: c for c in proj.discover()}
    if CELL_ID not in cells:
        raise RuntimeError(
            f"cell '{CELL_ID}' not found in {PROJECT}. "
            f"Available: {sorted(cells)}")
    c = cells[CELL_ID]
    z = float(c.params.get("z_split") or 0) or Z_SPLIT
    print(f"[qc] project {PROJECT}, cell {CELL_ID}, z_split {z}")

    # The region review happens between `audit` and `repair`, so repaired.swc
    # usually does not exist yet. Fall back to the raw input skeleton, which is
    # the right thing to look at anyway: the node ids in regions.csv refer to
    # that file.
    if c.repaired_swc.exists():
        swc = str(c.repaired_swc)
        print(f"[qc] loaded the REPAIRED skeleton: {swc}")
    elif c.swc and Path(c.swc).exists():
        swc = str(c.swc)
        print(f"[qc] repaired.swc not found - loaded the RAW input skeleton:")
        print(f"[qc]   {swc}")
        print(f"[qc] This is correct for reviewing regions.csv. Re-run this "
              f"script after `pipeline.py repair` to see the result.")
    else:
        raise RuntimeError(
            f"cell {CELL_ID} has neither {c.repaired_swc} nor a raw input SWC. "
            f"Run `pipeline.py discover` and check the paths.")
    return (swc,
            str(c.synapses) if c.synapses else "",
            str(c.regions_csv),
            str(c.branches_csv),
            z,
            str(c.mesh) if c.mesh else "")


def main():
    global SWC, SYNAPSE_CSV, REGIONS_CSV, BRANCHES_CSV, Z_SPLIT, MESH_DAE
    (SWC, SYNAPSE_CSV, REGIONS_CSV, BRANCHES_CSV, Z_SPLIT,
     MESH_DAE) = resolve_paths()
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    try:
        scene.unit_settings.length_unit = "MICROMETERS"
    except TypeError:
        pass          # older Blender: leave as metres, values are still um

    if not SWC:
        raise RuntimeError(
            "No skeleton to load. Set PROJECT and CELL_ID in the CONFIG block "
            "at the top, or set SWC to an explicit path.")
    if not os.path.exists(SWC):
        raise RuntimeError(f"SWC not found: {SWC}")

    pos, rad, typ, par = load_swc(SWC)
    roots = [i for i, p in par.items() if p == -1]
    soma = pos[min(roots)] if roots else Vector((0, 0, 0))
    offset = soma.copy() if RECENTRE else Vector((0, 0, 0))
    print(f"[qc] {len(pos)} nodes, soma at {tuple(round(v,2) for v in soma)}")

    paths = unbranched_paths(pos, par)
    print(f"[qc] {len(paths)} branches")

    # ---- the generated mesh, for validating the skeleton against -------
    try:
        if LOAD_MESH and MESH_DAE and os.path.exists(MESH_DAE):
            mv, mt = load_dae(MESH_DAE)
            zs = [v[2] for v in mv]
            print(f"[qc] mesh {len(mv)} verts, {len(mt)} tris, "
                  f"Z {min(zs):.2f}..{max(zs):.2f}")
            sz = [pos[i].z for i in pos]
            print(f"[qc] skeleton Z {min(sz):.2f}..{max(sz):.2f}"
                  + ("   <-- FRAME MISMATCH, check z_scale"
                     if (max(zs) - min(zs)) > 0 and
                        not 0.5 < (max(sz) - min(sz)) / (max(zs) - min(zs)) < 2.0
                     else "   (frames agree)"))
            me = bpy.data.meshes.new("Mesh_dae")
            me.from_pydata([tuple(Vector(v) - offset) for v in mv], [], mt)
            me.update()
            mob = bpy.data.objects.new("Mesh_dae", me)
            mmat = material("mat_mesh", (0.55, 0.60, 0.70, MESH_ALPHA))
            for attr, val in (("blend_method", "BLEND"),
                              ("surface_render_method", "BLENDED")):
                try:
                    setattr(mmat, attr, val)
                except (AttributeError, TypeError):
                    pass
            me.materials.append(mmat)
            collection("Mesh").objects.link(mob)
            if MESH_DECIMATE and 0 < MESH_DECIMATE < 1:
                d = mob.modifiers.new("Decimate", "DECIMATE")
                d.ratio = MESH_DECIMATE
                print(f"[qc] decimating mesh to {MESH_DECIMATE:.0%}")
            print(f"[qc] press Alt+Z for X-ray to see the skeleton inside it")
        elif LOAD_MESH and MESH_DAE:
            print(f"[qc] mesh not found: {MESH_DAE}")
        elif LOAD_MESH:
            print(f"[qc] no mesh for this cell - check mesh_dir and [patterns]")
    except Exception as e:
        print(f"[qc] MESH SKIPPED: {e}")

    root_coll = collection("Skeleton")
    groups = {"lobular": [], "arboreal": []}
    for path in paths:
        pts = [pos[n] for n in path]
        zmean = sum(p.z for p in pts) / len(pts)
        groups["arboreal" if zmean >= Z_SPLIT else "lobular"].append(pts)
    for comp, cols in (("lobular", (0.85, 0.55, 0.15, 1.0)),
                       ("arboreal", (0.30, 0.55, 0.85, 1.0))):
        if groups[comp]:
            make_curve(f"Skeleton_{comp}", groups[comp], root_coll,
                       material(f"mat_{comp}", cols), offset, SKEL_RADIUS)
            print(f"[qc]   {comp}: {len(groups[comp])} branches")

    # soma marker
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=16, ring_count=8,
        radius=max(rad.get(min(roots), 1.0), 0.5),
        location=tuple(soma - offset))
    s = bpy.context.active_object
    s.name = "Soma"
    for c in list(s.users_collection):
        c.objects.unlink(s)
    root_coll.objects.link(s)
    s.data.materials.append(material("mat_soma", (0.55, 0.55, 0.55, 1.0)))

    # ---- synapses -----------------------------------------------------
    # Wrapped: the Review markers below are the point of the pre-repair pass,
    # so a problem with the synapse CSV must not abort before they are built.
    try:
      if SYNAPSE_CSV and os.path.exists(SYNAPSE_CSV):
        syn_coll = collection("Synapses")
        with open(SYNAPSE_CSV, newline="") as fh:
            rdr = csv.DictReader(fh)
            cx = sniff(rdr.fieldnames, "x")
            cy = sniff(rdr.fieldnames, "y")
            cz = sniff(rdr.fieldnames, "z")
            ct = sniff(rdr.fieldnames, "type")
            if not (cx and cy and cz):
                raise RuntimeError(
                    f"could not find x/y/z columns in {SYNAPSE_CSV}\n"
                    f"       header is {rdr.fieldnames}")
            buckets = {}
            for row in rdr:
                try:
                    p = Vector((float(row[cx]), float(row[cy]), float(row[cz])))
                except (TypeError, ValueError):
                    continue
                label = (row.get(ct) or "unspecified").strip() if ct else "unspecified"
                buckets.setdefault(label, []).append(p)
        seen = {}
        for label, pts in sorted(buckets.items()):
            safe = "".join(c if c.isalnum() else "_" for c in label)[:40]
            make_points(f"Syn_{safe}", pts, syn_coll,
                        material(f"mat_syn_{safe}", colour_for(label, seen)),
                        offset, SYN_RADIUS)
            print(f"[qc] synapses {label}: {len(pts)}")
    except Exception as e:
        print(f"[qc] SYNAPSES SKIPPED: {e}")

    # ---- review markers ------------------------------------------------
    try:
        if REGIONS_CSV and os.path.exists(REGIONS_CSV):
            rev = collection("Review")
            n = 0
            with open(REGIONS_CSV, newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        p = Vector((float(row["x"]), float(row["y"]),
                                    float(row["z"])))
                    except (KeyError, TypeError, ValueError):
                        continue
                    e = bpy.data.objects.new(
                        f"REVIEW_r{row['region']}_"
                        f"{row.get('action','').strip()}", None)
                    e.empty_display_type = "SPHERE"
                    e.empty_display_size = 2.0
                    e.location = p - offset
                    rev.objects.link(e)
                    n += 1
            print(f"[qc] {n} review markers - select one, press Numpad-period")

            # ---- exactly which nodes each merge would delete -------------
            plan = collection("MergePlan")
            mk = material("mat_keep", (0.20, 0.70, 0.35, 1.0))
            md = material("mat_drop", (0.90, 0.20, 0.20, 1.0))
            mh = material("mat_hairpin", (0.95, 0.65, 0.10, 1.0))
            mo = material("mat_offmesh", (0.95, 0.10, 0.85, 1.0))
            ms = material("mat_spike", (0.95, 0.35, 0.05, 1.0))
            mf = material("mat_fragment", (0.20, 0.85, 0.90, 1.0))
            nk = nd = nh = no = ns = nf = 0
            with open(REGIONS_CSV, newline="") as fh:
                for row in csv.DictReader(fh):
                    rid = row.get("region", "?")
                    kind = (row.get("kind") or "duplicate").strip()
                    def chain(field):
                        out = []
                        for tok in (row.get(field) or "").split():
                            try:
                                k = int(tok)
                            except ValueError:
                                continue
                            if k in pos:
                                out.append(pos[k])
                        return out
                    if kind == "fragment":
                        pts = chain("node_ids")
                        if len(pts) >= 2:
                            make_curve(f"FRAGMENT_r{rid}", [pts], plan, mf,
                                       offset, SKEL_RADIUS * 2.0)
                        elif len(pts) == 1:
                            me3 = bpy.data.meshes.new(f"FRAGMENT_r{rid}")
                            me3.from_pydata([tuple(pts[0] - offset)], [], [])
                            me3.update()
                            fo = bpy.data.objects.new(f"FRAGMENT_r{rid}", me3)
                            fo.instance_type = "VERTS"
                            plan.objects.link(fo)
                            bpy.ops.mesh.primitive_ico_sphere_add(
                                subdivisions=2, radius=SYN_RADIUS * 1.4,
                                location=(0, 0, 0))
                            fm = bpy.context.active_object
                            fm.name = f"FRAGMENT_r{rid}_marker"
                            for cc in list(fm.users_collection):
                                cc.objects.unlink(fm)
                            plan.objects.link(fm)
                            fm.data.materials.clear()
                            fm.data.materials.append(mf)
                            fm.parent = fo
                        nf += 1
                        continue
                    if kind == "hairpin":
                        pts = chain("node_ids")
                        if len(pts) >= 2:
                            make_curve(f"HAIRPIN_r{rid}", [pts], plan, mh,
                                       offset, SKEL_RADIUS * 2.2)
                            nh += 1
                        continue
                    if kind in ("offmesh", "spike"):
                        ctx = chain("node_ids")
                        bad = chain("bad_ids")
                        mat = mo if kind == "offmesh" else ms
                        tag = "OFFMESH" if kind == "offmesh" else "SPIKE"
                        if len(ctx) >= 2:
                            make_curve(f"{tag}_r{rid}_context", [ctx], plan,
                                       mat, offset, SKEL_RADIUS * 1.4)
                        for j, p in enumerate(bad):
                            me2 = bpy.data.meshes.new(f"{tag}_r{rid}_bad{j}")
                            me2.from_pydata([tuple(p - offset)], [], [])
                            me2.update()
                            ho = bpy.data.objects.new(
                                f"{tag}_r{rid}_bad{j}", me2)
                            ho.instance_type = "VERTS"
                            plan.objects.link(ho)
                            bpy.ops.mesh.primitive_ico_sphere_add(
                                subdivisions=2, radius=SYN_RADIUS * 1.6,
                                location=(0, 0, 0))
                            mk2 = bpy.context.active_object
                            mk2.name = f"{tag}_r{rid}_marker{j}"
                            for cc in list(mk2.users_collection):
                                cc.objects.unlink(mk2)
                            plan.objects.link(mk2)
                            mk2.data.materials.clear()
                            mk2.data.materials.append(mat)
                            mk2.parent = ho
                        if kind == "offmesh":
                            no += 1
                        else:
                            ns += 1
                        continue
                    keep, drop = chain("keep_ids"), chain("drop_ids")
                    if len(keep) >= 2:
                        make_curve(f"KEEP_r{rid}", [keep], plan, mk,
                                   offset, SKEL_RADIUS * 1.8)
                        nk += 1
                    if len(drop) >= 2:
                        make_curve(f"DROP_r{rid}", [drop], plan, md,
                                   offset, SKEL_RADIUS * 2.6)
                        nd += 1
            print(f"[qc] MergePlan: {nk} KEEP (green), {nd} DROP (red), "
                  f"{nf} FRAGMENT (cyan), {nh} HAIRPIN (orange), "
                  f"{no} OFFMESH (magenta), {ns} SPIKE (orange-red)")
            print(f"[qc]   red/magenta is what disappears. Hide the Skeleton "
                  f"collection to see it clearly.")
            print(f"[qc]   OFFMESH and SPIKE mark the bad nodes with spheres; "
                  f"the thin curve is context.")
        elif REGIONS_CSV:
            print(f"[qc] no regions.csv at {REGIONS_CSV} - run "
                  f"`pipeline.py audit` first")
    except Exception as e:
        print(f"[qc] REVIEW MARKERS SKIPPED: {e}")

    # ---- optional density colouring ------------------------------------
    try:
      if BRANCHES_CSV and os.path.exists(BRANCHES_CSV):
        with open(BRANCHES_CSV, newline="") as fh:
            rows = list(csv.DictReader(fh))
        vals = []
        for r in rows:
            try:
                vals.append((int(r["branch"]), float(r["density_per_um"])))
            except (KeyError, TypeError, ValueError):
                continue
        if vals:
            dens = dict(vals)
            ordered = sorted(v for _, v in vals)
            cuts = [ordered[int(len(ordered) * f)] for f in (0.2, 0.4, 0.6, 0.8)]
            dcoll = collection("Density")
            bins = {i: [] for i in range(5)}
            for bi, path in enumerate(paths):
                d = dens.get(bi)
                if d is None:
                    continue
                k = sum(1 for c in cuts if d > c)
                bins[k].append([pos[n] for n in path])
            ramp = [(0.90, 0.94, 0.98, 1.0), (0.71, 0.83, 0.96, 1.0),
                    (0.52, 0.72, 0.92, 1.0), (0.22, 0.54, 0.87, 1.0),
                    (0.09, 0.37, 0.65, 1.0)]
            for k, pls in bins.items():
                if pls:
                    make_curve(f"Density_q{k+1}", pls, dcoll,
                               material(f"mat_dens_{k}", ramp[k]),
                               offset, SKEL_RADIUS)
            print(f"[qc] density quintile cuts: "
                  f"{[round(c,4) for c in cuts]} per um")
    except Exception as e:
        print(f"[qc] DENSITY COLOURING SKIPPED: {e}")

    print("[qc] done. Set viewport shading colour to Material to see the colours.")


main()
