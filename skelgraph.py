"""
Graph logic for the Blender skeleton round trip. No bpy, so it is testable.

Blender's mesh edit mode is the right tool for fixing a skeleton by hand: move
a node with G, delete with X, join two nodes with F, dissolve, merge with M,
extrude with E. But a mesh is an undirected edge soup and SWC needs a rooted
tree, so the conversion back has real work to do and real failure modes.

rebuild_tree() does that conversion and refuses rather than guessing when the
edit is not a tree.
"""

import collections
import math


class NotATree(Exception):
    """The edited mesh cannot be expressed as an SWC tree."""


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def components(n_verts, edges):
    """Connected components as a list of vertex-index lists."""
    adj = collections.defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen = set()
    out = []
    for v in range(n_verts):
        if v in seen:
            continue
        stack, comp = [v], []
        seen.add(v)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        out.append(sorted(comp))
    return out


def find_cycles(n_verts, edges):
    """Return a list of vertices where a cycle closes. Empty means acyclic.

    A tree on n vertices has exactly n-1 edges per component, so a component
    with more edges than that contains a cycle. Reported per component rather
    than as the full cycle, which is enough to navigate to it in Blender.
    """
    adj = collections.defaultdict(set)
    for a, b in edges:
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
    bad = []
    seen = set()
    for comp in components(n_verts, edges):
        cset = set(comp)
        n_edges = sum(1 for a, b in edges
                      if a in cset and b in cset and a != b)
        n_edges = len({(min(a, b), max(a, b)) for a, b in edges
                       if a in cset and b in cset and a != b})
        if n_edges > len(comp) - 1:
            # walk it to find a vertex that closes a loop
            start = comp[0]
            parent = {start: None}
            stack = [start]
            seen.add(start)
            found = None
            while stack and found is None:
                x = stack.pop()
                for y in adj[x]:
                    if y == parent.get(x):
                        continue
                    if y in parent:
                        found = y
                        break
                    parent[y] = x
                    stack.append(y)
            bad.append(found if found is not None else start)
    return bad


def choose_root(comp, coords, types, radii, soma_hint=None, soma_type=1):
    """Pick the root of a component.

    Preference order: a node still typed as soma, then the node nearest a
    remembered soma position, then the thickest node. The last is a guess and
    the caller is told about it.
    """
    somas = [v for v in comp if types.get(v) == soma_type]
    if somas:
        return max(somas, key=lambda v: radii.get(v, 0.0)), 'soma type'
    if soma_hint is not None:
        return min(comp, key=lambda v: _dist(coords[v], soma_hint)), \
            'nearest to remembered soma'
    return max(comp, key=lambda v: radii.get(v, 0.0)), 'thickest node (guess)'


def fill_missing_radii(comp, edges_adj, radii, fallback=0.1):
    """Give new vertices a radius from their neighbours.

    Extrude and subdivide create vertices that carry no radius, and a zero
    radius would poison every downstream calculation that divides by it.
    """
    filled = {}
    unknown = [v for v in comp if not radii.get(v)]
    known = {v: radii[v] for v in comp if radii.get(v)}
    for v in unknown:
        vals = [known[n] for n in edges_adj[v] if n in known]
        if not vals:
            ring = {m for n in edges_adj[v] for m in edges_adj[n]}
            vals = [known[m] for m in ring if m in known]
        filled[v] = (sum(vals) / len(vals)) if vals else fallback
    return filled


def weld_points(polylines, tol=0.05, radii=None):
    """Turn a set of polylines into a welded vertex/edge graph.

    A Blender curve stores no topology. One spline per branch means a branch
    point exists only as the coincident endpoint of two or more splines, so
    rebuilding the tree starts by merging points that sit on top of each other.

    polylines  list of lists of (x, y, z)
    radii      matching list of lists of radius, or None
    tol        weld distance. Branch points are exact duplicates on import, so
               a small value suffices; raise it only if you have nudged a
               junction apart and want it rejoined.

    Returns (coords, edges, radii_out, report). Nodes take the largest radius
    of the points welded into them, since a junction belongs to the thicker
    process.
    """
    cell = max(tol, 1e-9)
    grid = {}
    coords, rad_out = {}, {}
    counts = collections.Counter()
    nxt = 0

    def key(p):
        return (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)),
                int(math.floor(p[2] / cell)))

    def find_or_make(p, r):
        nonlocal nxt
        kx, ky, kz = key(p)
        best, bestd = None, tol
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for v in grid.get((kx + dx, ky + dy, kz + dz), ()):
                        d = _dist(coords[v], p)
                        if d <= bestd:
                            best, bestd = v, d
        if best is not None:
            counts[best] += 1
            if r is not None:
                rad_out[best] = max(rad_out.get(best, 0.0), r)
            return best
        v = nxt
        nxt += 1
        coords[v] = tuple(p)
        if r is not None:
            rad_out[v] = r
        grid.setdefault((kx, ky, kz), []).append(v)
        counts[v] = 1
        return v

    edges = set()
    self_edges = 0
    for si, pl in enumerate(polylines):
        rs = radii[si] if radii else [None] * len(pl)
        ids = [find_or_make(p, r) for p, r in zip(pl, rs)]
        for a, b in zip(ids, ids[1:]):
            if a == b:
                self_edges += 1
                continue
            edges.add((min(a, b), max(a, b)))

    welded = sum(c - 1 for c in counts.values() if c > 1)
    report = dict(n_splines=len(polylines),
                  n_points=sum(len(p) for p in polylines),
                  n_nodes=len(coords), welded=welded,
                  junctions=sum(1 for c in counts.values() if c > 1),
                  collapsed_segments=self_edges)
    return coords, sorted(edges), rad_out, report


def rebuild_tree(coords, edges, radii=None, types=None, soma_hint=None,
                 soma_type=1, default_type=3, fallback_radius=0.1):
    """Turn an undirected mesh graph into SWC rows.

    coords  {vertex index: (x, y, z)}
    edges   iterable of (a, b) vertex index pairs
    radii   {vertex index: radius}, missing entries are interpolated
    types   {vertex index: swc type}, missing entries get default_type

    Returns (rows, report). rows are (id, type, x, y, z, radius, parent) with
    ids renumbered from 1 in parents-before-children order.

    Raises NotATree on a cycle, since every downstream tool assumes a tree and
    a cycle would send the path walkers into an infinite loop.
    """
    radii = dict(radii or {})
    types = dict(types or {})
    n = (max(coords) + 1) if coords else 0

    clean = {(min(a, b), max(a, b)) for a, b in edges if a != b}
    cyc = find_cycles(n, clean)
    if cyc:
        raise NotATree(
            f"{len(cyc)} cycle(s) in the edited mesh, near vertex "
            f"{', '.join(str(c) for c in cyc[:5])}. A skeleton has to be a "
            f"tree. In Blender, select the vertex and delete one of its edges.")

    adj = collections.defaultdict(list)
    for a, b in clean:
        adj[a].append(b)
        adj[b].append(a)

    comps = [c for c in components(n, clean) if c]
    comps.sort(key=len, reverse=True)
    report = dict(n_vertices=len(coords), n_edges=len(clean),
                  n_components=len(comps), roots=[], filled_radii=0,
                  isolated=0, notes=[])

    rows = []
    next_id = 1
    for ci, comp in enumerate(comps):
        if len(comp) == 1 and not adj[comp[0]]:
            report['isolated'] += 1
            continue
        root, how = choose_root(comp, coords, types, radii,
                                soma_hint if ci == 0 else None,
                                soma_type=soma_type)
        report['roots'].append(dict(component=ci, size=len(comp),
                                    vertex=root, chosen_by=how))
        if how.endswith('(guess)'):
            report['notes'].append(
                f"component {ci} ({len(comp)} nodes) had no soma-typed node; "
                f"rooted at the thickest node instead")

        fills = fill_missing_radii(comp, adj, radii,
                                   fallback=fallback_radius)
        report['filled_radii'] += len(fills)
        local = dict(radii)
        local.update(fills)

        order, parent = [], {root: -1}
        stack = [root]
        seen = {root}
        while stack:
            v = stack.pop(0)
            order.append(v)
            for w in sorted(adj[v]):
                if w not in seen:
                    seen.add(w)
                    parent[w] = v
                    stack.append(w)

        idmap = {}
        for v in order:
            idmap[v] = next_id
            next_id += 1
        for v in order:
            x, y, z = coords[v]
            rows.append((idmap[v],
                         types.get(v, soma_type if v == root and
                                   types.get(v) == soma_type else default_type),
                         x, y, z,
                         local.get(v, fallback_radius),
                         idmap[parent[v]] if parent[v] != -1 else -1))

    if report['isolated']:
        report['notes'].append(
            f"{report['isolated']} isolated vertex/vertices dropped - a node "
            f"with no edges carries no path length")
    if len(report['roots']) > 1:
        report['notes'].append(
            f"{len(report['roots'])} separate components written as separate "
            f"roots. The density stage wants one tree, so join them in Blender "
            f"(select two nodes, press F) or delete the strays.")
    return rows, report


def write_swc(path, rows, header=()):
    with open(path, 'w') as fh:
        for h in header:
            fh.write(f"# {h}\n")
        fh.write("\n")
        for i, t, x, y, z, r, p in rows:
            fh.write(f"{i} {t} {x:.4f} {y:.4f} {z:.4f} {r:.4f} {p}\n")


def total_length(rows):
    pos = {i: (x, y, z) for i, t, x, y, z, r, p in rows}
    return sum(_dist(pos[i], pos[p])
               for i, t, x, y, z, r, p in rows if p != -1)
