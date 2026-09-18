bl_info = {
    "name": "Skeleton Editor (SWC)",
    "author": "PLA Pipeline",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar (N) > Skeleton",
    "description": "Edit neuron skeletons as curves and write them back to SWC",
    "category": "Import-Export",
}

"""
Edit a neuron skeleton by hand in Blender and write it back to SWC.

The skeleton comes in as a CURVE - one poly spline per unbranched branch - so
you edit it with ordinary curve tools. Export welds coincident spline endpoints
back into junctions and rebuilds the rooted tree.

INSTALL: see README.md in this directory, or Edit > Preferences > Add-ons >
Install from Disk. Set Tools Directory in the add-on preferences to the
repository root - the folder holding skelgraph.py and cellpaths.py, one level
above this one. Export needs skelgraph.py.

CURVE EDITING THAT MAPS ONTO SKELETON REPAIR
  G                     move a node back onto the centreline
  X > Vertices          delete a node: a spike, a stray, a whole twig
  E                     extrude, to extend a branch cut short
  F                     join two selected endpoints into one spline
  right-click Subdivide add nodes along a segment
  Alt+S                 shrink/fatten - edits the radius, which is exported
  L / Ctrl+L            select a whole spline; L then X deletes a branch
  X > Segment           break a spline in two

  A curve control point cannot have three neighbours, so a junction exists only
  as coincident endpoints. To MOVE a branch point: select the child branch's
  endpoint, turn on vertex snapping (Shift+Tab, Snap To > Vertex), and G it onto
  the parent point you want. Snapping matters - Weld Tolerance is 0.02 um by
  default, so eyeballing it leaves the branch detached.
"""

import csv
import os

import bpy
from bpy.props import (BoolProperty, FloatProperty, PointerProperty,
                       StringProperty)
from mathutils import Vector


# ============================================================== pure helpers

def read_swc(path):
    pos, rad, typ, par = {}, {}, {}, {}
    for line in open(path):
        if line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 7:
            continue
        i = int(f[0])
        typ[i] = int(f[1])
        pos[i] = (float(f[2]), float(f[3]), float(f[4]))
        rad[i] = float(f[5])
        par[i] = int(f[6])
    return pos, rad, typ, par


def load_dae(path, to_volume_coords=True):
    """Parse a Viking COLLADA export into (verts, tris).

    Deliberately not bpy.ops.wm.collada_import: that applies the file's <unit>
    tag, which in a Viking export is the source volume's pixel resolution
    rather than a real scale factor, and would land the mesh in a different
    frame from the skeleton.
    """
    import re
    data = open(path, "r", errors="replace").read()
    pm = re.search(r'positions-array" count="(\d+)">(.*?)</float_array>',
                   data, re.S)
    tm = re.search(r'<triangles count="(\d+)".*?<p>(.*?)</p>', data, re.S)
    if not (pm and tm):
        raise RuntimeError(f"no geometry found in {path}")
    flat = [float(v) for v in pm.group(2).split()]
    verts = [tuple(flat[i:i + 3]) for i in range(0, len(flat), 3)]
    idx = [int(v) for v in tm.group(2).split()]
    tris = [tuple(idx[i:i + 3]) for i in range(0, len(idx), 3)]
    if to_volume_coords:
        nm = re.search(r"<translate>([^<]+)</translate>", data)
        if nm:
            ox, oy, oz = (float(v) for v in nm.group(1).split())
            verts = [(x + ox, y + oy, z + oz) for x, y, z in verts]
    return verts, tris


def unbranched_paths(pos, par):
    """One list per branch: root/branch-point to branch-point or tip."""
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


def coll(name):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


def mat(name, rgba):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    m = bpy.data.materials.new(name)
    try:
        if not getattr(m, "use_nodes", True):
            m.use_nodes = True
    except (AttributeError, TypeError):
        pass
    tree = getattr(m, "node_tree", None)
    b = tree.nodes.get("Principled BSDF") if tree else None
    if b:
        b.inputs["Base Color"].default_value = rgba
    m.diffuse_color = rgba
    return m


def gather_curve(ob):
    """Polylines and radii off the curve, in volume coordinates."""
    off = Vector(ob.get("offset", (0, 0, 0)))
    pls, rds = [], []
    for sp in ob.data.splines:
        pts = list(sp.bezier_points) if sp.type == "BEZIER" else list(sp.points)
        if not pts:
            continue
        pl, rd = [], []
        for p in pts:
            co = p.co
            v = Vector((co[0], co[1], co[2])) + off
            pl.append((v.x, v.y, v.z))
            rd.append(float(getattr(p, "radius", 1.0)))
        pls.append(pl)
        rds.append(rd)
    return pls, rds


def abspath(p):
    return bpy.path.abspath(p) if p else ""


def prefs():
    try:
        return bpy.context.preferences.addons[__name__].preferences
    except (KeyError, AttributeError):
        return None


def need_module(name):
    """Import a pipeline module, or explain exactly what is missing."""
    import importlib
    import sys
    pr = prefs()
    tdir = abspath(pr.tools_dir) if pr else ""
    if tdir and tdir not in sys.path:
        sys.path.insert(0, tdir)
    importlib.invalidate_caches()
    try:
        return __import__(name)
    except ImportError:
        present = []
        if tdir and os.path.isdir(tdir):
            present = sorted(f for f in os.listdir(tdir) if f.endswith(".py"))
        raise RuntimeError(
            f"cannot import {name}.py\n"
            f"  Tools Directory: {tdir or '(not set)'}\n"
            f"  .py files there: {', '.join(present) if present else 'none'}\n"
            f"  Set it in Edit > Preferences > Add-ons > Skeleton Editor, and "
            f"make sure {name}.py is in that folder.") from None


# ==================================================================== import

def do_import(st, report=print, force=False):
    swc = abspath(st.swc_path)
    if not swc:
        raise RuntimeError("pick an SWC file first")
    if not os.path.exists(swc):
        raise RuntimeError(f"SWC not found: {swc}")

    name = st.object_name or f"Skel_{os.path.splitext(os.path.basename(swc))[0]}"
    existing = bpy.data.objects.get(name)
    if existing is not None and not force:
        n_sp = len(existing.data.splines) if existing.type == "CURVE" else 0
        report(f"{name} is already in the scene ({n_sp} splines) - not touching "
               f"it. Your edits are safe.")
        report("Use Re-import to discard them and reload from disk.")
        return existing

    pos, rad, typ, par = read_swc(swc)
    report(f"loaded {len(pos)} nodes from {os.path.basename(swc)}")

    roots = [i for i, p in par.items() if p == -1]
    soma = Vector(pos[min(roots)]) if roots else Vector((0, 0, 0))
    offset = soma.copy() if st.recentre else Vector((0, 0, 0))

    branches = unbranched_paths(pos, par)
    report(f"{len(branches)} branches, {sum(len(b) for b in branches)} control "
           f"points (junctions appear in every spline that meets there)")

    if existing is not None:
        report(f"discarding the existing {name} and reloading")
        bpy.data.objects.remove(existing, do_unlink=True)

    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 1.0 if st.true_calibre else st.line_radius
    cu.bevel_resolution = 2 if st.true_calibre else 1
    for b in branches:
        sp = cu.splines.new("POLY")
        sp.points.add(len(b) - 1)
        for k, nid in enumerate(b):
            q = Vector(pos[nid]) - offset
            sp.points[k].co = (q.x, q.y, q.z, 1.0)
            sp.points[k].radius = rad[nid]   # stored whatever the display setting
    ob = bpy.data.objects.new(name, cu)
    cu.materials.append(mat("mat_skel_edit", (0.85, 0.70, 0.25, 1.0)))
    coll("SkeletonEdit").objects.link(ob)

    order = sorted(pos)
    ob["swc_source"] = swc
    ob["soma_hint"] = list(soma)
    ob["offset"] = list(offset)
    ob["orig_xyz"] = [c for n in order for c in pos[n]]
    ob["orig_type"] = [typ[n] for n in order]
    ob["orig_rad"] = [rad[n] for n in order]

    shortest = min((Vector(pos[n]) - Vector(pos[par[n]])).length
                   for n in order if par[n] != -1) if len(order) > 1 else 0.0
    ob["shortest_segment"] = shortest
    report(f"shortest segment {shortest:.4f} um; weld tolerance "
           f"{st.weld_tol} um"
           + ("" if st.weld_tol < 0.5 * shortest else
              "  <-- TOO LARGE, it will merge adjacent nodes"))

    # ---- reference: the two analysis boundaries --------------------------
    ref = coll("Reference")
    if st.show_zsplit and st.z_split:
        span = max(max(p[0] for p in pos.values()) - min(p[0] for p in pos.values()),
                   max(p[1] for p in pos.values()) - min(p[1] for p in pos.values())) * 1.2
        bpy.ops.mesh.primitive_plane_add(size=span,
                                         location=(0, 0, st.z_split - offset.z))
        pl = bpy.context.active_object
        pl.name = f"ZSplit_{st.z_split:.2f}um"
        for cc in list(pl.users_collection):
            cc.objects.unlink(pl)
        ref.objects.link(pl)
        pm = mat("mat_zsplit", (0.30, 0.80, 0.45, 0.18))
        for attr, val in (("blend_method", "BLEND"),
                          ("surface_render_method", "BLENDED")):
            try:
                setattr(pm, attr, val)
            except (AttributeError, TypeError):
                pass
        pl.data.materials.append(pm)
        pl.hide_select = True
        lob = sum(1 for p in pos.values() if p[2] < st.z_split)
        report(f"z_split {st.z_split:.3f}: {lob} nodes below (lobular), "
               f"{len(pos)-lob} above (arboreal)")

    if st.show_soma and roots:
        rootid = min(roots)
        somas = [i for i in pos if typ[i] == 1]
        srad = max(rad[i] for i in somas) if somas else rad[rootid]
        excl = srad if st.soma_exclusion < 0 else st.soma_exclusion
        if excl > 0:
            e = bpy.data.objects.new(f"SomaExclusion_{excl:.2f}um", None)
            e.empty_display_type = "SPHERE"
            e.empty_display_size = excl
            e.location = tuple(Vector(pos[rootid]) - offset)
            ref.objects.link(e)
            e.hide_select = True
            report(f"soma exclusion sphere {excl:.3f} um")

    # ---- flagged regions -------------------------------------------------
    rp = abspath(st.regions_path)
    if rp and os.path.exists(rp):
        try:
            fc = coll("Flagged")
            made = 0
            with open(rp, newline="") as fh:
                for row in csv.DictReader(fh):
                    kind = (row.get("kind") or "duplicate").strip()
                    tag = {"duplicate": "dup", "fragment": "frag",
                           "hairpin": "hairpin", "offmesh": "offmesh",
                           "spike": "spike"}.get(kind, kind)
                    col = {"dup": (0.90, 0.20, 0.20, 1.0),
                           "frag": (0.20, 0.85, 0.90, 1.0),
                           "hairpin": (0.95, 0.65, 0.10, 1.0),
                           "offmesh": (0.95, 0.10, 0.85, 1.0),
                           "spike": (0.95, 0.35, 0.05, 1.0)}.get(
                               tag, (1, 1, 1, 1))
                    ids = []
                    for field in ("bad_ids", "drop_ids", "node_ids"):
                        toks = (row.get(field) or "").split()
                        if toks:
                            ids = [int(t) for t in toks if t.isdigit()]
                            break
                    pts = [Vector(pos[i]) - offset for i in ids if i in pos]
                    if not pts:
                        continue
                    fname = f"{tag}_{row.get('region', '?')}"
                    fcu = bpy.data.curves.new(fname, "CURVE")
                    fcu.dimensions = "3D"
                    fcu.bevel_depth = max(st.line_radius * 2.5, 0.12)
                    fsp = fcu.splines.new("POLY")
                    if len(pts) >= 2:
                        fsp.points.add(len(pts) - 1)
                        for k, q in enumerate(pts):
                            fsp.points[k].co = (q.x, q.y, q.z, 1.0)
                    else:
                        fsp.points.add(1)
                        q = pts[0]
                        fsp.points[0].co = (q.x, q.y, q.z, 1.0)
                        fsp.points[1].co = (q.x, q.y, q.z + 0.35, 1.0)
                    fcu.materials.append(mat(f"mat_flag_{tag}", col))
                    fob = bpy.data.objects.new(fname, fcu)
                    fc.objects.link(fob)
                    fob.hide_select = True
                    made += 1
            report(f"{made} flagged region(s), unselectable reference only")
        except Exception as e:
            report(f"flagged regions skipped: {e}")

    # ---- reference mesh --------------------------------------------------
    mp = abspath(st.mesh_path)
    if st.load_mesh and mp and os.path.exists(mp):
        try:
            mv, mt = load_dae(mp)
            zs = [v[2] for v in mv]
            sz = [p[2] for p in pos.values()]
            agree = 0.5 < (max(sz) - min(sz)) / max(max(zs) - min(zs), 1e-9) < 2.0
            report(f"mesh {len(mv)} verts, Z {min(zs):.2f}..{max(zs):.2f}; "
                   f"skeleton Z {min(sz):.2f}..{max(sz):.2f} "
                   f"{'(frames agree)' if agree else '<-- FRAME MISMATCH'}")
            if "Mesh_ref" in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects["Mesh_ref"],
                                        do_unlink=True)
            rme = bpy.data.meshes.new("Mesh_ref")
            rme.from_pydata([tuple(Vector(v) - offset) for v in mv], [], mt)
            rme.update()
            rob = bpy.data.objects.new("Mesh_ref", rme)
            mm = mat("mat_mesh_ref", (0.55, 0.60, 0.70, st.mesh_alpha))
            for attr, val in (("blend_method", "BLEND"),
                              ("surface_render_method", "BLENDED")):
                try:
                    setattr(mm, attr, val)
                except (AttributeError, TypeError):
                    pass
            rme.materials.append(mm)
            coll("SkeletonEdit").objects.link(rob)
            rob.hide_select = True
            report("reference mesh is unselectable; Alt+Z for X-ray")
        except Exception as e:
            report(f"mesh skipped: {e}")

    st.object_name = name
    report(f"Tab into Edit Mode on {name}")
    return ob


# ==================================================================== export

def do_export(st, validate_only=False, report=print):
    skelgraph = need_module("skelgraph")

    ob = bpy.data.objects.get(st.object_name) if st.object_name else None
    if ob is None or ob.type != "CURVE" or not ob.get("swc_source"):
        ob = bpy.context.object
    if ob is None or ob.type != "CURVE" or not ob.get("swc_source"):
        cand = [o for o in bpy.data.objects
                if o.type == "CURVE" and o.get("swc_source")]
        if len(cand) != 1:
            raise RuntimeError("select the skeleton curve first, or set the "
                               "Object field in the panel")
        ob = cand[0]
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")      # flush edit-mode changes

    pls, rds = gather_curve(ob)
    shortest = float(ob.get("shortest_segment", 0.0) or 0.0)
    if shortest and st.weld_tol >= 0.5 * shortest:
        report(f"WARNING: weld tolerance {st.weld_tol} is at least half the "
               f"shortest original segment ({shortest:.4f} um) - it will merge "
               f"nodes that should stay separate")

    coords, edges, radii, wrep = skelgraph.weld_points(pls, tol=st.weld_tol,
                                                       radii=rds)
    report(f"{wrep['n_splines']} splines, {wrep['n_points']} points -> "
           f"{wrep['n_nodes']} nodes ({wrep['welded']} welded at "
           f"{wrep['junctions']} junctions)")
    if wrep["collapsed_segments"]:
        report(f"  {wrep['collapsed_segments']} zero-length segment(s) "
               f"collapsed by welding")

    types = {}
    oxyz = list(ob.get("orig_xyz", []))
    otyp = list(ob.get("orig_type", []))
    orad = list(ob.get("orig_rad", []))
    if oxyz and otyp:
        pts = [(oxyz[i], oxyz[i + 1], oxyz[i + 2])
               for i in range(0, len(oxyz), 3)]
        cell = max(st.weld_tol, 1e-6)
        grid = {}
        for n, p in enumerate(pts):
            grid.setdefault((int(p[0] // cell), int(p[1] // cell),
                             int(p[2] // cell)), []).append(n)
        matched = 0
        for v, p in coords.items():
            best, bd = None, st.weld_tol
            kx, ky, kz = (int(p[0] // cell), int(p[1] // cell),
                          int(p[2] // cell))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for n in grid.get((kx + dx, ky + dy, kz + dz), ()):
                            q = pts[n]
                            d = ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
                                 + (p[2] - q[2]) ** 2) ** 0.5
                            if d <= bd:
                                best, bd = n, d
            if best is not None:
                types[v] = otyp[best]
                matched += 1
                if orad and abs(radii.get(v, 1.0) - 1.0) < 1e-9:
                    radii[v] = orad[best]
        report(f"  matched {matched}/{len(coords)} nodes to original positions "
               f"for type and radius")

    stray = [v for v in coords
             if abs(radii.get(v, 1.0) - 1.0) < 1e-9 and v not in types]
    for v in stray:
        radii.pop(v, None)          # let rebuild_tree interpolate instead
    if stray:
        report(f"  {len(stray)} new point(s) had Blender's default radius 1.0; "
               f"interpolating from neighbours instead")

    hint = tuple(ob.get("soma_hint", (0, 0, 0)))
    try:
        rows, rep = skelgraph.rebuild_tree(coords, edges, radii, types,
                                          soma_hint=hint)
    except skelgraph.NotATree as e:
        report(f"NOT EXPORTED: {e}")
        return None

    report(f"{len(rows)} nodes, {len(rep['roots'])} root(s), "
           f"{skelgraph.total_length(rows):.1f} um")
    for r in rep["roots"]:
        report(f"  component {r['component']}: {r['size']} nodes, root chosen "
               f"by {r['chosen_by']}")
    if rep["filled_radii"]:
        report(f"  interpolated a radius for {rep['filled_radii']} node(s)")
    for n in rep["notes"]:
        report(f"  note: {n}")

    if validate_only:
        report("validate only, nothing written")
        return rows

    out = abspath(st.out_path)
    if not out:
        raise RuntimeError("set the Export To path in the panel")
    d = os.path.dirname(out)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    skelgraph.write_swc(out, rows, header=[
        "EDITED by hand in Blender via the Skeleton Editor add-on",
        f"source {ob.get('swc_source')}",
        f"welded at {st.weld_tol} um",
        "node ids renumbered - re-run `pipeline.py audit` before using an old "
        "regions.csv",
    ])
    report(f"wrote {out}")
    return rows


# ================================================================ properties

class SKEL_Settings(bpy.types.PropertyGroup):
    swc_path: StringProperty(
        name="SWC", subtype="FILE_PATH",
        description="The skeleton to load. Point it at the raw export, an "
                    "edited.swc or a cleaned.swc - whichever you want to work "
                    "on or just look at")
    mesh_path: StringProperty(
        name="Mesh (.dae)", subtype="FILE_PATH",
        description="Viking COLLADA export, loaded as unselectable reference "
                    "so you can see the skeleton inside the membrane")
    regions_path: StringProperty(
        name="regions.csv", subtype="FILE_PATH",
        description="Optional. Draws what the detectors flagged, as "
                    "unselectable reference")
    out_path: StringProperty(
        name="Export To", subtype="FILE_PATH",
        description="Where Export writes. Directories are created if needed")
    object_name: StringProperty(
        name="Object", default="",
        description="Which curve Export reads. Set by Import")

    recentre: BoolProperty(
        name="Soma at origin", default=True,
        description="Shift everything so the soma sits at the world origin. "
                    "Volume coordinates are restored on export")
    load_mesh: BoolProperty(name="Load mesh", default=True)
    mesh_alpha: FloatProperty(name="Mesh alpha", default=0.15, min=0.0, max=1.0)
    true_calibre: BoolProperty(
        name="True calibre", default=False,
        description="Draw the curve at each node's real radius. Off gives a "
                    "thin line, which is what you want for editing points. "
                    "Either way the radius is stored and exported")
    line_radius: FloatProperty(name="Line radius", default=0.04, min=0.0,
                               max=5.0)
    weld_tol: FloatProperty(
        name="Weld tolerance", default=0.02, min=0.0, max=1.0, precision=4,
        description="On export, points closer than this weld into one node - "
                    "that is how junctions are rebuilt. Keep it well under the "
                    "shortest segment")
    show_zsplit: BoolProperty(name="Show z-split", default=True)
    z_split: FloatProperty(
        name="z_split", default=0.0, precision=3,
        description="Compartment boundary in volume Z. 0 hides the plane")
    show_soma: BoolProperty(name="Show soma sphere", default=True)
    soma_exclusion: FloatProperty(
        name="Soma exclusion", default=-1.0, precision=3,
        description="-1 uses the soma node's own radius, 0 hides it")


class SKEL_Prefs(bpy.types.AddonPreferences):
    bl_idname = __name__

    tools_dir: StringProperty(
        name="Tools Directory", subtype="DIR_PATH",
        description="Folder containing skelgraph.py and cellpaths.py. Export "
                    "needs skelgraph.py")
    project_dir: StringProperty(
        name="Default Project", subtype="DIR_PATH",
        description="Optional. A pipeline project directory, used by "
                    "'Fill from project'")

    def draw(self, context):
        self.layout.prop(self, "tools_dir")
        self.layout.prop(self, "project_dir")
        self.layout.label(
            text="Export imports skelgraph.py from Tools Directory.",
            icon="INFO")


# ================================================================= operators

def _run(op, fn):
    try:
        fn(report=lambda m: (print(f"[skel] {m}"), op.report({'INFO'}, m)))
    except Exception as e:
        for line in str(e).splitlines():
            print(f"[skel] {line}")
        op.report({'ERROR'}, str(e).splitlines()[0])
        return {'CANCELLED'}
    return {'FINISHED'}


class SKEL_OT_import(bpy.types.Operator):
    bl_idname = "skel.import_swc"
    bl_label = "Import"
    bl_description = "Load the SWC as an editable curve. Does nothing if it is "\
                     "already in the scene"

    def execute(self, context):
        st = context.scene.skel
        return _run(self, lambda report: do_import(st, report=report))


class SKEL_OT_reimport(bpy.types.Operator):
    bl_idname = "skel.reimport_swc"
    bl_label = "Re-import (discards edits)"
    bl_description = "Reload from disk, throwing away everything in the "\
                     "viewport"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        st = context.scene.skel
        return _run(self, lambda report: do_import(st, report=report,
                                                   force=True))


class SKEL_OT_validate(bpy.types.Operator):
    bl_idname = "skel.validate_swc"
    bl_label = "Validate"
    bl_description = "Check the edit is still a tree. Writes nothing"

    def execute(self, context):
        st = context.scene.skel
        return _run(self, lambda report: do_export(st, validate_only=True,
                                                   report=report))


class SKEL_OT_export(bpy.types.Operator):
    bl_idname = "skel.export_swc"
    bl_label = "Export"
    bl_description = "Write the edited skeleton to the Export To path"

    def execute(self, context):
        st = context.scene.skel
        return _run(self, lambda report: do_export(st, report=report))


class SKEL_OT_suggest_out(bpy.types.Operator):
    bl_idname = "skel.suggest_out"
    bl_label = "Beside input"
    bl_description = "Set Export To next to the SWC, as <name>_edited.swc, so "\
                     "the input is never overwritten"

    def execute(self, context):
        st = context.scene.skel
        src = abspath(st.swc_path)
        if not src:
            self.report({'ERROR'}, "pick an SWC first")
            return {'CANCELLED'}
        base, ext = os.path.splitext(src)
        st.out_path = base + "_edited" + (ext or ".swc")
        return {'FINISHED'}


class SKEL_OT_fill_from_project(bpy.types.Operator):
    bl_idname = "skel.fill_from_project"
    bl_label = "Fill from project"
    bl_description = "Fill the paths, z_split and soma exclusion from a "\
                     "pipeline project's cells.toml for one cell"

    cell_id: StringProperty(name="Cell ID", default="192")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        st = context.scene.skel
        pr = prefs()
        root = abspath(pr.project_dir) if pr else ""
        if not root:
            self.report({'ERROR'},
                        "set Default Project in the add-on preferences")
            return {'CANCELLED'}
        try:
            cellpaths = need_module("cellpaths")
            proj = cellpaths.Project(root)
            cells = {c.cell_id: c for c in proj.discover()}
            if self.cell_id not in cells:
                raise RuntimeError(f"cell '{self.cell_id}' not in {root}; "
                                   f"have {sorted(cells)}")
            c = cells[self.cell_id]
            src = (c.edited_swc if c.edited_swc.exists()
                   else (c.repaired_swc if c.repaired_swc.exists() else c.swc))
            st.swc_path = str(src)
            st.mesh_path = str(c.mesh) if c.mesh else ""
            st.regions_path = str(c.regions_csv)
            st.out_path = str(c.edited_swc)
            st.z_split = float(c.params.get("z_split") or 0.0)
            st.soma_exclusion = float(c.params.get("exclude_soma_radius", -1.0))
            print(f"[skel] filled from {root}, cell {self.cell_id}: "
                  f"{os.path.basename(str(src))}")
        except Exception as e:
            for line in str(e).splitlines():
                print(f"[skel] {line}")
            self.report({'ERROR'}, str(e).splitlines()[0])
            return {'CANCELLED'}
        return {'FINISHED'}


# ===================================================================== panels

class SKEL_PT_files(bpy.types.Panel):
    bl_label = "Skeleton"
    bl_idname = "SKEL_PT_files"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Skeleton"

    def draw(self, context):
        st = context.scene.skel
        lay = self.layout

        box = lay.box()
        box.label(text="Input", icon="FILE_FOLDER")
        box.prop(st, "swc_path")
        box.prop(st, "mesh_path")
        box.prop(st, "regions_path")
        box.operator("skel.fill_from_project", icon="PRESET")

        box = lay.box()
        box.label(text="Output", icon="EXPORT")
        box.prop(st, "out_path")
        box.operator("skel.suggest_out", icon="DUPLICATE")

        col = lay.column(align=True)
        col.scale_y = 1.3
        col.operator("skel.import_swc", icon="IMPORT")
        col.separator()
        col.operator("skel.validate_swc", icon="CHECKMARK")
        col.operator("skel.export_swc", icon="EXPORT")
        lay.operator("skel.reimport_swc", icon="FILE_REFRESH")

        try:
            ob = bpy.data.objects.get(st.object_name) or context.object
            if ob and ob.type == "CURVE" and ob.get("swc_source"):
                b = lay.box()
                b.label(text=ob.name, icon="OUTLINER_OB_CURVE")
                sp = ob.data.splines
                b.label(text=f"{len(sp)} splines")
                b.label(text=f"{sum(len(s.points) if s.type != 'BEZIER' else len(s.bezier_points) for s in sp)} points")
        except Exception:
            pass


class SKEL_PT_options(bpy.types.Panel):
    bl_label = "Options"
    bl_idname = "SKEL_PT_options"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Skeleton"
    bl_parent_id = "SKEL_PT_files"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        st = context.scene.skel
        lay = self.layout
        lay.prop(st, "weld_tol")
        lay.prop(st, "recentre")
        lay.separator()
        lay.prop(st, "true_calibre")
        if not st.true_calibre:
            lay.prop(st, "line_radius")
        lay.prop(st, "load_mesh")
        if st.load_mesh:
            lay.prop(st, "mesh_alpha")
        lay.separator()
        lay.prop(st, "show_zsplit")
        if st.show_zsplit:
            lay.prop(st, "z_split")
        lay.prop(st, "show_soma")
        if st.show_soma:
            lay.prop(st, "soma_exclusion")
        lay.prop(st, "object_name")


CLASSES = (SKEL_Settings, SKEL_Prefs,
           SKEL_OT_import, SKEL_OT_reimport, SKEL_OT_validate, SKEL_OT_export,
           SKEL_OT_suggest_out, SKEL_OT_fill_from_project,
           SKEL_PT_files, SKEL_PT_options)


def register():
    for c in CLASSES:
        try:
            bpy.utils.register_class(c)
        except Exception:
            try:
                bpy.utils.unregister_class(c)
            except Exception:
                pass
            bpy.utils.register_class(c)
    bpy.types.Scene.skel = PointerProperty(type=SKEL_Settings)


def unregister():
    try:
        del bpy.types.Scene.skel
    except Exception:
        pass
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass


if __name__ == "__main__":
    register()
    print("[skel] registered. Press N in the 3D view, 'Skeleton' tab.")
