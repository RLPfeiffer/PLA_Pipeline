"""
swcrepair - repair centroid-derived SWC skeletons from serial-section EM annotation.

Designed for SBFSEM-pytools / Viking output where the skeleton is built from
per-section contour centroids. Fixes, in the order they must be applied:

  1. soma collapse        - a chain of large-radius nodes -> one node
  2. duplicate merge      - one wide process traced as two parallel branches
  3. orphan handling      - graft or drop disconnected fragments
  4. multifurcation split - k>2 children -> nested bifurcations
  5. smooth + resample    - remove per-section centroid jitter

Topology is always fixed before geometry: smoothing a duplicated region just
produces two nicely smoothed phantom branches.
"""

import collections
import numpy as np
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.interpolate import splprep, splev


# ----------------------------------------------------------------- I/O

class Skel:
    """Mutable SWC skeleton. ids are arbitrary ints; renumber() makes them 1..n."""

    def __init__(self):
        self.pos = {}      # id -> np.array([x,y,z])
        self.rad = {}      # id -> float
        self.typ = {}      # id -> int
        self.par = {}      # id -> int (-1 for root)
        self.header = []

    # -- construction ------------------------------------------------

    @classmethod
    def load(cls, path):
        s = cls()
        for line in open(path):
            if line.startswith('#'):
                s.header.append(line.rstrip('\n'))
                continue
            f = line.split()
            if len(f) < 7:
                continue
            i = int(f[0])
            s.typ[i] = int(f[1])
            s.pos[i] = np.array([float(f[2]), float(f[3]), float(f[4])])
            s.rad[i] = float(f[5])
            s.par[i] = int(f[6])
        return s

    def save(self, path, extra_header=()):
        self.renumber()
        with open(path, 'w') as fh:
            for h in self.header:
                fh.write(h + '\n')
            for h in extra_header:
                fh.write('# ' + h + '\n')
            fh.write('\n')
            for i in sorted(self.pos):
                p = self.pos[i]
                fh.write(f"{i} {self.typ[i]} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} "
                         f"{self.rad[i]:.4f} {self.par[i]}\n")

    def copy(self):
        s = Skel()
        s.pos = {i: p.copy() for i, p in self.pos.items()}
        s.rad = dict(self.rad)
        s.typ = dict(self.typ)
        s.par = dict(self.par)
        s.header = list(self.header)
        return s

    # -- structure ---------------------------------------------------

    def children(self):
        ch = collections.defaultdict(list)
        for i, p in self.par.items():
            if p != -1:
                ch[p].append(i)
        return ch

    def roots(self):
        return [i for i, p in self.par.items() if p == -1]

    def depths(self):
        """Path length from each node back to its root."""
        ch = self.children()
        d = {}
        for r in self.roots():
            st = [(r, 0.0)]
            while st:
                n, acc = st.pop()
                d[n] = acc
                for c in ch[n]:
                    st.append((c, acc + float(np.linalg.norm(self.pos[c] - self.pos[n]))))
        return d

    def ancestors(self, i):
        out = []
        cur = i
        seen = set()
        while cur != -1 and cur not in seen:
            out.append(cur)
            seen.add(cur)
            cur = self.par.get(cur, -1)
        return out

    def tree_dist(self, a, b, depth):
        A = set(self.ancestors(a))
        for cur in self.ancestors(b):
            if cur in A:
                return depth[a] + depth[b] - 2 * depth[cur]
        return float('inf')

    def paths(self, min_nodes=1):
        """Unbranched runs: root/branch-point -> next branch-point or tip.

        Both endpoints are included, so adjacent paths share their junction
        node. A path start is a root or a node with >= 2 children.
        """
        ch = self.children()
        starts = [i for i in self.pos if self.par[i] == -1 or len(ch[i]) >= 2]
        out = []
        for s in starts:
            for c in ch[s]:
                p = [s, c]
                while len(ch[p[-1]]) == 1:
                    p.append(ch[p[-1]][0])
                if len(p) >= min_nodes:
                    out.append(p)
        return out

    def total_length(self):
        return sum(float(np.linalg.norm(self.pos[i] - self.pos[p]))
                   for i, p in self.par.items() if p != -1)

    def stats(self):
        ch = self.children()
        nch = {i: len(ch[i]) for i in self.pos}
        return dict(nodes=len(self.pos),
                    roots=len(self.roots()),
                    tips=sum(1 for v in nch.values() if v == 0),
                    bifurcations=sum(1 for v in nch.values() if v == 2),
                    multifurcations=sum(1 for v in nch.values() if v > 2),
                    length=self.total_length())

    def check(self):
        """Return list of structural problems. Empty list == valid tree."""
        bad = []
        for i, p in self.par.items():
            if p != -1 and p not in self.pos:
                bad.append(f"node {i} has missing parent {p}")
        for i in self.pos:
            seen = set()
            cur = i
            while cur != -1:
                if cur in seen:
                    bad.append(f"cycle through node {i}")
                    break
                seen.add(cur)
                cur = self.par.get(cur, -1)
                if len(seen) > len(self.pos):
                    bad.append(f"runaway chain from {i}")
                    break
        return bad

    def renumber(self):
        """Reassign ids 1..n in parents-before-children order."""
        ch = self.children()
        new = {}
        nxt = 1
        for r in sorted(self.roots()):
            st = [r]
            while st:
                n = st.pop(0)
                new[n] = nxt
                nxt += 1
                st.extend(sorted(ch[n]))
        # anything unreachable (shouldn't happen) keeps its place at the end
        for i in sorted(self.pos):
            if i not in new:
                new[i] = nxt
                nxt += 1
        self.pos = {new[i]: v for i, v in self.pos.items()}
        self.rad = {new[i]: v for i, v in self.rad.items()}
        self.typ = {new[i]: v for i, v in self.typ.items()}
        self.par = {new[i]: (new[p] if p != -1 else -1) for i, p in self.par.items()}

    # -- editing primitives ------------------------------------------

    def delete(self, ids, reparent_to=None):
        """Delete nodes. Orphaned children are reparented per reparent_to map,
        or to the deleted node's parent if not specified."""
        ids = set(ids)
        ch = self.children()
        for i in ids:
            for c in ch[i]:
                if c in ids:
                    continue
                tgt = (reparent_to or {}).get(i, self.par[i])
                # walk up out of the deleted set
                while tgt in ids:
                    tgt = self.par.get(tgt, -1)
                self.par[c] = tgt
        for i in ids:
            self.pos.pop(i, None)
            self.rad.pop(i, None)
            self.typ.pop(i, None)
            self.par.pop(i, None)

    def new_id(self):
        return (max(self.pos) + 1) if self.pos else 1


# ----------------------------------------- 0. strip appended markers

def strip_types(skel, types=(6,), verbose=True):
    """Remove nodes of the given SWC types entirely.

    Use for types that are appended annotation markers rather than traced
    neurite. In SBFSEM-pytools output, type 6 nodes are single-node stubs
    hanging off a cable node at roughly one parent-radius offset - i.e. they
    sit on the membrane, not on the centreline - and they are almost all
    terminal. Leaving them in inflates path length and fabricates tips.
    """
    drop = [i for i in skel.pos if skel.typ[i] in types]
    if not drop:
        return 0
    before = skel.total_length()
    n_tips = sum(1 for i in drop if not any(p == i for p in skel.par.values()))
    skel.delete(drop)
    if verbose:
        print(f"  stripped {len(drop)} nodes of type(s) {types} "
              f"({n_tips} were tips), removing {before - skel.total_length():.1f} um")
    return len(drop)


def marker_diagnosis(skel, t):
    """Evidence that nodes of type t are membrane markers, not centreline.

    Returns offset/parent-radius statistics: a ratio near 1 with high
    correlation means the nodes sit on the membrane of their parent.
    """
    ns = [i for i in skel.pos if skel.typ[i] == t and skel.par[i] != -1]
    if len(ns) < 5:
        return None
    ch = skel.children()
    off = np.array([np.linalg.norm(skel.pos[i] - skel.pos[skel.par[i]]) for i in ns])
    pr = np.array([skel.rad[skel.par[i]] for i in ns])
    ok = pr > 1e-6
    return dict(n=len(ns),
                tip_fraction=sum(1 for i in ns if len(ch[i]) == 0) / len(ns),
                median_offset=float(np.median(off)),
                median_parent_radius=float(np.median(pr)),
                offset_over_radius=float(np.median(off[ok] / pr[ok])),
                correlation=float(np.corrcoef(off[ok], pr[ok])[0, 1]),
                length_carried=float(off.sum()))


# ------------------------------------------------- 1. soma collapse

def collapse_soma(skel, soma_type=1, verbose=True):
    """Replace a chain of soma-typed nodes with a single node at their centroid."""
    soma = [i for i in skel.pos if skel.typ[i] == soma_type]
    if len(soma) <= 1:
        return skel, 0
    ch = skel.children()
    centre = np.mean([skel.pos[i] for i in soma], axis=0)
    radius = max(skel.rad[i] for i in soma)

    keep = min(soma)                      # reuse the lowest id as the soma node
    skel.pos[keep] = centre
    skel.rad[keep] = radius
    skel.typ[keep] = soma_type
    skel.par[keep] = -1

    drop = [i for i in soma if i != keep]
    reparent = {i: keep for i in drop}
    skel.delete(drop, reparent_to=reparent)
    if verbose:
        print(f"  soma: {len(soma)} nodes -> 1  at {centre.round(3)} r={radius:.3f}")
    return skel, len(drop)


def ch_of(skel, i):
    return [c for c, p in skel.par.items() if p == i]


# --------------------------------------- 2. duplicate detection/merge

def find_duplicates(skel, max_sep=1.6, radius_slack=0.4, radius_factor=1.2,
                    min_tree_dist=6.0, cluster_gap=2.0, min_pairs=4):
    """Find regions where one process was traced as two parallel branches.

    Signature: node pairs close in space but far apart along the tree.
    Returns list of dicts with the node sets and summary geometry.
    """
    ids = list(skel.pos)
    arr = np.array([skel.pos[i] for i in ids])
    depth = skel.depths()
    hits = []
    for a, b in cKDTree(arr).query_pairs(r=max_sep, output_type='ndarray'):
        ia, ib = ids[a], ids[b]
        e = float(np.linalg.norm(arr[a] - arr[b]))
        if e > radius_factor * (skel.rad[ia] + skel.rad[ib]) + radius_slack:
            continue
        td = skel.tree_dist(ia, ib, depth)
        if td > min_tree_dist:
            hits.append((ia, ib, e, td))
    if not hits:
        return []

    mids = np.array([(skel.pos[a] + skel.pos[b]) / 2 for a, b, _, _ in hits])
    lab = fcluster(linkage(mids, 'single'), t=cluster_gap, criterion='distance')

    groups = collections.defaultdict(list)
    for h, L in zip(hits, lab):
        groups[L].append(h)

    out = []
    for L, hs in groups.items():
        if len(hs) < min_pairs:
            continue
        nodes = sorted({x for h in hs for x in h[:2]})
        seg = sum(float(np.linalg.norm(skel.pos[i] - skel.pos[skel.par[i]]))
                  for i in nodes if skel.par.get(i, -1) in skel.pos)
        tds = [h[3] for h in hs]
        out.append(dict(
            region=int(L),
            pairs=[(h[0], h[1]) for h in hs],
            nodes=nodes,
            centroid=np.mean([(skel.pos[a] + skel.pos[b]) / 2 for a, b, _, _ in hs], axis=0),
            mean_radius=float(np.mean([(skel.rad[a] + skel.rad[b]) / 2 for a, b, _, _ in hs])),
            length_involved=float(seg),
            median_tree_dist=float(np.median([t for t in tds if np.isfinite(t)]) if any(np.isfinite(tds)) else np.inf),
            separate_components=bool(not any(np.isfinite(t) for t in tds)),
        ))
    return sorted(out, key=lambda d: -d['length_involved'])


def merge_plan(skel, region):
    """What merge_region WOULD do, without touching anything.

    Returns dict(keep=[ids], drop=[ids], reparent={drop_id: keep_id},
    moved={keep_id: new position}) or None if the region would be a no-op.

    Exactly mirrors merge_region's decisions, so reviewing the plan is
    reviewing the merge. Use this to see which nodes disappear before agreeing
    to anything.
    """
    depth = skel.depths()
    nodes = [i for i in region['nodes'] if i in skel.pos]
    if len(nodes) < 4:
        return None
    path_of = {}
    for pi, p in enumerate(skel.paths()):
        for n in p:
            path_of.setdefault(n, pi)
    counts = collections.Counter(path_of.get(n, -1) for n in nodes)
    if len(counts) < 2:
        return None
    (pa, _), (pb, _) = counts.most_common(2)
    A = [n for n in nodes if path_of.get(n) == pa]
    B = [n for n in nodes if path_of.get(n) == pb]
    if not A or not B:
        return None
    if (len(A), -np.mean([depth.get(n, 0) for n in A])) < \
       (len(B), -np.mean([depth.get(n, 0) for n in B])):
        A, B = B, A

    partners = collections.defaultdict(list)
    for a, b in region['pairs']:
        if a in A and b in B:
            partners[a].append(b)
        elif b in A and a in B:
            partners[b].append(a)
    moved = {}
    for a, ps in partners.items():
        ps = [p for p in ps if p in skel.pos]
        if ps:
            moved[a] = (skel.pos[a]
                        + np.mean([skel.pos[p] for p in ps], axis=0)) / 2

    tree = cKDTree(np.array([skel.pos[a] for a in A]))
    reparent = {}
    for b in B:
        _, k = tree.query(skel.pos[b])
        reparent[b] = A[int(k)]
    return dict(keep=sorted(A), drop=sorted(B), reparent=reparent, moved=moved,
                keep_length=_chain_length(skel, A),
                drop_length=_chain_length(skel, B))


def _chain_length(skel, ids):
    s = 0.0
    idset = set(ids)
    for i in ids:
        p = skel.par.get(i, -1)
        if p in idset:
            s += float(np.linalg.norm(skel.pos[i] - skel.pos[p]))
    return s


def find_hairpins(skel, min_angle=120.0, min_run=2, window=3, min_length=0.3):
    """Find places where a path reverses direction - a hairpin.

    Different signature from find_duplicates: a hairpin is a single chain that
    folds back on itself, so the two limbs can be far enough apart that the
    proximity test never fires, and close enough along the tree that the
    topological-distance test rejects them too. This looks at turning angle
    instead.

    min_angle   degrees of direction change that counts as a reversal
    min_run     consecutive reversing steps needed (1 is noise)
    window      nodes either side used to estimate local direction; larger
                values ignore per-section jitter
    """
    out = []
    for path in skel.paths():
        if len(path) < 2 * window + 2:
            continue
        P = np.array([skel.pos[n] for n in path])
        run = []
        for k in range(window, len(P) - window):
            a = P[k] - P[k - window]
            b = P[k + window] - P[k]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9:
                continue
            cos = float(np.clip(np.dot(a, b) / (na * nb), -1, 1))
            ang = np.degrees(np.arccos(cos))
            if ang >= min_angle:
                run.append(k)
            else:
                if len(run) >= min_run:
                    out.append((path, run))
                run = []
        if len(run) >= min_run:
            out.append((path, run))

    regions = []
    for path, run in out:
        lo = max(0, run[0] - window)
        hi = min(len(path) - 1, run[-1] + window)
        seg = path[lo:hi + 1]
        L = _chain_length(skel, seg)
        if L < min_length:
            continue
        P = np.array([skel.pos[n] for n in seg])
        chord = float(np.linalg.norm(P[-1] - P[0]))
        regions.append(dict(
            nodes=seg,
            pairs=[],
            centroid=P.mean(axis=0),
            mean_radius=float(np.median([skel.rad[n] for n in seg])),
            length_involved=L,
            chord=chord,
            tortuosity=(L / chord) if chord > 1e-6 else float('inf'),
            n_reversals=len(run),
            separate_components=False,
        ))
    regions.sort(key=lambda d: -d['tortuosity'])
    for n, d in enumerate(regions, 1):
        d['region'] = 1000 + n          # distinct id space from find_duplicates
    return regions


def find_spikes(skel, window=4, factor=4.0, floor=0.75, min_run=1, pad=1):
    """Find nodes that jump off the local course of their own branch.

    A spike is a short excursion - often one or two nodes - that shoots away
    from the process and comes straight back. The hairpin detector smooths over
    a window to ignore per-section jitter, which also blunts a one-node spike,
    and the duplicate detector never sees it because there is no second chain.
    So this compares each node against a moving average of its neighbours and
    flags anything further out than factor * local radius.

    factor  multiples of the local radius that count as a spike
    floor   minimum deviation in um, so thin processes do not over-trigger
    pad     nodes either side included in the reported region
    """
    out = []
    for path in skel.paths():
        if len(path) < 2 * window + 3:
            continue
        P = np.array([skel.pos[n] for n in path])
        sm = P.copy()
        for k in range(len(P)):
            a, b = max(0, k - window), min(len(P), k + window + 1)
            idx = [j for j in range(a, b) if j != k]
            sm[k] = P[idx].mean(axis=0)
        dev = np.linalg.norm(P - sm, axis=1)
        run = []
        for k in range(1, len(P) - 1):
            r = max(skel.rad[path[k]], 0.01)
            if dev[k] > max(factor * r, floor):
                run.append(k)
            else:
                if len(run) >= min_run:
                    out.append((path, run, dev))
                run = []
        if len(run) >= min_run:
            out.append((path, run, dev))

    regions = []
    for path, run, dev in out:
        lo = max(0, run[0] - pad)
        hi = min(len(path) - 1, run[-1] + pad)
        seg = path[lo:hi + 1]
        P = np.array([skel.pos[n] for n in seg])
        regions.append(dict(
            nodes=seg, pairs=[], bad=[path[k] for k in run],
            centroid=P.mean(axis=0),
            mean_radius=float(np.median([skel.rad[n] for n in seg])),
            length_involved=_chain_length(skel, seg),
            max_deviation=float(dev[run].max()),
            n_spike_nodes=len(run),
            separate_components=False,
        ))
    regions.sort(key=lambda d: -d['max_deviation'])
    for n, d in enumerate(regions, 1):
        d['region'] = 2000 + n
    return regions


def find_offmesh(skel, verts, factor=2.5, floor=0.5, pad=1):
    """Find skeleton nodes that sit far from the mesh they came from.

    The strongest check available, because the mesh is ground truth: a
    centreline node should be roughly one local radius from the nearest
    membrane vertex. A node much further out has left the process.

    verts   mesh vertices in the SAME frame as the skeleton - use
            load_dae_mesh(), which applies the node <translate>
    factor  multiples of the local radius that count as off-mesh
    """
    tree = cKDTree(np.asarray(verts))
    flagged = {}
    for i in skel.pos:
        d, _ = tree.query(skel.pos[i])
        r = max(skel.rad[i], 0.01)
        if d > max(factor * r, floor):
            flagged[i] = float(d)
    if not flagged:
        return []

    regions = []
    for path in skel.paths():
        run = []
        for k, n in enumerate(path):
            if n in flagged:
                run.append(k)
            else:
                if run:
                    regions.append((path, run))
                run = []
        if run:
            regions.append((path, run))

    out = []
    for path, run in regions:
        lo = max(0, run[0] - pad)
        hi = min(len(path) - 1, run[-1] + pad)
        seg = path[lo:hi + 1]
        P = np.array([skel.pos[n] for n in seg])
        out.append(dict(
            nodes=seg, pairs=[], bad=[path[k] for k in run],
            centroid=P.mean(axis=0),
            mean_radius=float(np.median([skel.rad[n] for n in seg])),
            length_involved=_chain_length(skel, seg),
            max_distance=max(flagged[path[k]] for k in run),
            n_off_nodes=len(run),
            separate_components=False,
        ))
    out.sort(key=lambda d: -d['max_distance'])
    for n, d in enumerate(out, 1):
        d['region'] = 3000 + n
    return out


def prune_short_branches(skel, factor=1.0, min_abs=0.0, max_passes=2000,
                         verbose=True):
    """Remove branches shorter than factor x their own diameter.

    A branch shorter than the process is thick is not a branch: it is a surface
    bump or a contour-linker artefact. Terminal branches are deleted. Internal
    ones are DISSOLVED - the branch's nodes go and the junction at its far end
    moves back onto the junction at its near end - because deleting an internal
    branch outright would orphan everything downstream.

    Runs to a fixed point, one branch per pass, since removing a terminal
    branch can leave its parent both terminal and short.
    """
    removed_terminal = dissolved_internal = 0
    removed_length = 0.0
    for _ in range(max_passes):
        ch = skel.children()
        target = None
        for path in skel.paths():
            ids = [n for n in path if n in skel.pos]
            if len(ids) < 2:
                continue
            L = _chain_length(skel, ids)
            r = float(np.median([skel.rad[n] for n in ids]))
            if L >= max(factor * 2 * r, min_abs):
                continue
            # nothing may BRANCH off the middle. Every mid-branch node has
            # exactly one child by construction, so the test is >1, not
            # truthiness - which silently blocked every branch of 3+ nodes.
            if any(len(ch[n]) > 1 for n in ids[1:-1]):
                continue
            target = (ids, L, bool(ch[ids[-1]]))
            break
        if target is None:
            break
        ids, L, internal = target
        start, end = ids[0], ids[-1]
        if internal:
            # move end's children onto start, then delete the whole branch
            # after the start node. This genuinely merges the two junctions.
            doomed = ids[1:]
            skel.delete(doomed, reparent_to={n: start for n in doomed})
            dissolved_internal += 1
        else:
            skel.delete(ids[1:])
            removed_terminal += 1
        removed_length += L
    if verbose:
        print(f"  removed {removed_terminal} short terminal branch(es), "
              f"dissolved {dissolved_internal} short internal branch(es), "
              f"{removed_length:.2f} um")
    return dict(removed_terminal=removed_terminal,
                dissolved_internal=dissolved_internal,
                removed_length=removed_length)


def drop_zero_length(skel, tol=0.005, verbose=True):
    """Collapse zero-length segments, which are pure bookkeeping.

    split_multifurcations inserts a near-zero connector stub to turn a k>2
    junction into nested bifurcations. Those stubs appear in the branch table
    as phantom branches with no length. Anything else of zero length is a
    duplicate node.
    """
    n = 0
    while True:
        victim = None
        for i in list(skel.pos):
            p = skel.par.get(i, -1)
            if p == -1:
                continue
            if float(np.linalg.norm(skel.pos[i] - skel.pos[p])) < tol:
                victim = i
                break
        if victim is None:
            break
        skel.delete([victim], reparent_to={victim: skel.par[victim]})
        n += 1
    if verbose and n:
        print(f"  collapsed {n} zero-length segment(s)")
    return n


def split_fragments(regions):
    """Separate true duplicates from disconnected fragments.

    find_duplicates uses 'close in space, far apart along the tree', and an
    infinite tree distance satisfies that trivially - so a fragment lying
    alongside the main tree looks exactly like a duplicated process. It is not
    one. Merging it would delete a chain; what it needs is joining on, or
    dropping. Returns (duplicates, fragments).
    """
    dups = [d for d in regions if not d.get('separate_components')]
    frags = [d for d in regions if d.get('separate_components')]
    for n, d in enumerate(frags, 1):
        d['region'] = 4000 + n
        d['kind'] = 'fragment'
    return dups, frags


def describe_fragment(skel, region):
    """Which nodes are stranded, and how wide the gap is on each side."""
    def root_of(n):
        cur = n
        while skel.par.get(cur, -1) != -1:
            cur = skel.par[cur]
        return cur

    sizes = collections.Counter(root_of(n) for n in skel.pos)
    main = max(sizes, key=lambda r: sizes[r])
    stranded = [n for n in region['nodes'] if root_of(n) != main]
    if not stranded:
        return None
    main_ids = [n for n in skel.pos if root_of(n) == main]
    tree = cKDTree(np.array([skel.pos[n] for n in main_ids]))
    gaps = []
    for n in stranded:
        d, k = tree.query(skel.pos[n])
        gaps.append((n, float(d), main_ids[int(k)]))
    frag_root = root_of(stranded[0])
    return dict(stranded=stranded,
                fragment_size=sizes[frag_root],
                nearest=min(gaps, key=lambda g: g[1]),
                gaps=gaps)


def graft_region(skel, region, max_gap=3.0, verbose=True):
    """Attach a stranded fragment to the nearest node of the main tree.

    This is the fix when the skeleton is simply broken - a gap with a node or
    two orphaned in it - rather than doubled. Refuses beyond max_gap, since a
    long graft invents a connection that may not exist.
    """
    info = describe_fragment(skel, region)
    if not info:
        return 0
    n, gap, target = info['nearest']
    if gap > max_gap:
        if verbose:
            print(f"  region {region.get('region')}: not grafted, nearest main "
                  f"tree node is {gap:.2f} um away (max_gap {max_gap})")
        return 0
    root = n
    while skel.par.get(root, -1) != -1:
        root = skel.par[root]
    skel.par[root] = target
    if verbose:
        print(f"  region {region.get('region')}: grafted fragment "
              f"({info['fragment_size']} node(s)) onto {target}, "
              f"gap {gap:.2f} um")
    return info['fragment_size']


def drop_region(skel, region, verbose=True):
    """Delete a stranded fragment outright."""
    info = describe_fragment(skel, region)
    if not info:
        return 0

    def root_of(n):
        cur = n
        while skel.par.get(cur, -1) != -1:
            cur = skel.par[cur]
        return cur

    frag_root = root_of(info['stranded'][0])
    ch = skel.children()
    doomed, st = [], [frag_root]
    while st:
        x = st.pop()
        doomed.append(x)
        st.extend(ch[x])
    before = skel.total_length()
    skel.delete(doomed)
    if verbose:
        print(f"  region {region.get('region')}: dropped {len(doomed)} "
              f"node(s), {before - skel.total_length():.2f} um removed")
    return len(doomed)


def snip_region(skel, region, verbose=True):
    """Delete the offending nodes of a spike or off-mesh region.

    Only the nodes the detector actually flagged are removed; the padding
    either side stays. Children of a removed node are reattached to its
    surviving parent, so a mid-branch excursion becomes a straight connection
    and a flagged tip simply disappears.
    """
    bad = [n for n in region.get('bad', []) if n in skel.pos]
    if not bad:
        return 0
    before = skel.total_length()
    skel.delete(bad)
    if verbose:
        print(f"  region {region.get('region')}: snipped {len(bad)} node(s), "
              f"{before - skel.total_length():.2f} um removed")
    return len(bad)


def trim_region(skel, region, verbose=True):
    """Delete the fold in a hairpin and connect straight across it.

    The flagged segment is a contiguous run of nodes. The interior is removed
    and the two ends joined, so the path length drops to roughly the chord.
    This throws away real annotation, so only use it once you have looked at
    the region and decided the fold is a tracing error rather than anatomy.
    """
    seg = [n for n in region['nodes'] if n in skel.pos]
    if len(seg) < 3:
        return 0
    first, last = seg[0], seg[-1]
    interior = [n for n in seg[1:-1]]
    if not interior:
        return 0
    # only safe if the interior carries nothing else
    ch = skel.children()
    for n in interior:
        for c in ch[n]:
            if c not in seg:
                if verbose:
                    print(f"  region {region.get('region')}: not trimmed, "
                          f"node {n} has a branch hanging off it")
                return 0
    before = skel.total_length()
    skel.delete(interior, reparent_to={n: first for n in interior})
    skel.par[last] = first
    if verbose:
        print(f"  region {region.get('region')}: trimmed {len(interior)} node(s), "
              f"{before - skel.total_length():.2f} um removed")
    return len(interior)


def merge_region(skel, region, verbose=True):
    """Collapse a duplicated region into one chain.

    Keeps the branch whose nodes are shallower (closer to the soma), moves its
    nodes onto the midline between the two branches, deletes the duplicate
    branch's nodes in the region, and reparents anything hanging off them onto
    the nearest kept node.
    """
    depth = skel.depths()
    nodes = [i for i in region['nodes'] if i in skel.pos]
    if len(nodes) < 4:
        return 0

    # assign each region node to its unbranched path, keep the two biggest
    path_of = {}
    for pi, p in enumerate(skel.paths()):
        for n in p:
            path_of.setdefault(n, pi)
    counts = collections.Counter(path_of.get(n, -1) for n in nodes)
    if len(counts) < 2:
        return 0
    (pa, _), (pb, _) = counts.most_common(2)

    A = [n for n in nodes if path_of.get(n) == pa]
    B = [n for n in nodes if path_of.get(n) == pb]
    if not A or not B:
        return 0
    # keeper = the chain carrying more of the region's geometry; ties go to
    # whichever sits closer to the soma
    if (len(A), -np.mean([depth.get(n, 0) for n in A])) < \
       (len(B), -np.mean([depth.get(n, 0) for n in B])):
        A, B = B, A

    # pull keeper nodes onto the midline, and take the union radius
    partners = collections.defaultdict(list)
    for a, b in region['pairs']:
        if a in A and b in B:
            partners[a].append(b)
        elif b in A and a in B:
            partners[b].append(a)
    for a, ps in partners.items():
        ps = [p for p in ps if p in skel.pos]
        if not ps:
            continue
        skel.pos[a] = (skel.pos[a] + np.mean([skel.pos[p] for p in ps], axis=0)) / 2
        skel.rad[a] = max([skel.rad[a]] + [skel.rad[p] for p in ps])

    # reparent everything hanging off B onto the nearest kept node
    Aarr = np.array([skel.pos[a] for a in A])
    tree = cKDTree(Aarr)
    reparent = {}
    for b in B:
        _, k = tree.query(skel.pos[b])
        reparent[b] = A[int(k)]
    skel.delete(B, reparent_to=reparent)
    if verbose:
        print(f"  region {region['region']}: merged, dropped {len(B)} nodes "
              f"(kept {len(A)}) at Z={region['centroid'][2]:.2f}")
    return len(B)


# ------------------------------------------------- 3. orphan handling

def handle_orphans(skel, main_root=None, graft_within=2.0, drop_smaller_than=1,
                   verbose=True):
    """Graft disconnected fragments onto the main tree, or drop them."""
    ch = skel.children()

    def subtree(r):
        out, st = [], [r]
        while st:
            n = st.pop()
            out.append(n)
            st.extend(ch[n])
        return out

    roots = skel.roots()
    sizes = {r: len(subtree(r)) for r in roots}
    if main_root is None:
        main_root = max(sizes, key=lambda r: sizes[r])
    main = set(subtree(main_root))
    main_ids = sorted(main)
    tree = cKDTree(np.array([skel.pos[i] for i in main_ids]))

    grafted = dropped = 0
    for r in roots:
        if r == main_root:
            continue
        frag = subtree(r)
        d, k = tree.query(skel.pos[r])
        if d <= graft_within:
            skel.par[r] = main_ids[int(k)]
            grafted += 1
            if verbose:
                print(f"  orphan root {r} ({len(frag)} nodes) grafted to "
                      f"{main_ids[int(k)]} at {d:.3f} um")
        elif len(frag) <= drop_smaller_than:
            skel.delete(frag)
            dropped += 1
            if verbose:
                print(f"  orphan root {r} ({len(frag)} nodes) dropped, "
                      f"nearest main tree node {d:.3f} um away")
        else:
            if verbose:
                print(f"  orphan root {r} ({len(frag)} nodes) LEFT ALONE, "
                      f"nearest main tree node {d:.3f} um away - review")
    return grafted, dropped


# ----------------------------------------- 4. multifurcation splitting

def split_multifurcations(skel, offset=1e-3, verbose=True):
    """Turn k>2 children into nested bifurcations via near-zero-length links."""
    n = 0
    while True:
        ch = skel.children()
        target = next((i for i in skel.pos if len(ch[i]) > 2), None)
        if target is None:
            break
        kids = sorted(ch[target])
        keep, rest = kids[0], kids[1:]
        stub = skel.new_id()
        direction = np.mean([skel.pos[k] - skel.pos[target] for k in rest], axis=0)
        norm = np.linalg.norm(direction)
        direction = direction / norm * offset if norm > 0 else np.zeros(3)
        skel.pos[stub] = skel.pos[target] + direction
        skel.rad[stub] = skel.rad[target]
        skel.typ[stub] = skel.typ[target]
        skel.par[stub] = target
        for k in rest:
            skel.par[k] = stub
        n += 1
    if verbose and n:
        print(f"  split {n} multifurcation(s) into nested bifurcations")
    return n


# --------------------------------------------- 5. smooth and resample

def smooth_resample(skel, step=0.25, smooth_coef=0.8, min_nodes=4, verbose=True):
    """Fit a smoothing spline per unbranched path and resample at fixed arc length.

    smooth_coef scales the allowed residual against local radius: jitter is
    high-frequency relative to real dendritic curvature, and the natural
    length scale for genuine bending is the process radius.
    Branch points and tips are held fixed so topology is preserved exactly.
    """
    out = Skel()
    out.header = list(skel.header)
    # keep every branch point / tip / root, they anchor the topology
    ch = skel.children()
    anchors = {i for i in skel.pos
               if skel.par[i] == -1 or len(ch[i]) != 1}
    remap = {}
    for a in anchors:
        remap[a] = a
        out.pos[a] = skel.pos[a].copy()
        out.rad[a] = skel.rad[a]
        out.typ[a] = skel.typ[a]
        out.par[a] = -1                     # fixed up below
    nxt = max(skel.pos) + 1
    n_fit = n_raw = 0

    for path in skel.paths():
        P = np.array([skel.pos[n] for n in path])
        types = [skel.typ[n] for n in path]
        rads = [skel.rad[n] for n in path]
        start, end = path[0], path[-1]

        if len(path) < min_nodes:
            interior = path[1:-1]
            chain = [remap[start]]
            for n in interior:
                nid = nxt
                nxt += 1
                out.pos[nid] = skel.pos[n].copy()
                out.rad[nid] = skel.rad[n]
                out.typ[nid] = skel.typ[n]
                chain.append(nid)
            chain.append(remap[end])
            for a, b in zip(chain, chain[1:]):
                out.par[b] = a
            n_raw += 1
            continue

        r = float(np.median(rads))
        s_par = len(P) * (smooth_coef * max(r, 0.05)) ** 2
        try:
            tck, _ = splprep(P.T, s=s_par, k=min(3, len(P) - 1))
        except Exception:
            tck = None
        if tck is None:
            fine = P
        else:
            fine = np.array(splev(np.linspace(0, 1, 400), tck)).T
            fine[0], fine[-1] = P[0], P[-1]       # hold the anchors exactly
        d = float(np.linalg.norm(np.diff(fine, axis=0), axis=1).sum())
        k = max(2, int(round(d / step)) + 1)
        if tck is None:
            res = P
        else:
            res = np.array(splev(np.linspace(0, 1, k), tck)).T
            res[0], res[-1] = P[0], P[-1]

        mode_type = collections.Counter(types[1:-1] or types).most_common(1)[0][0]
        chain = [remap[start]]
        for q in res[1:-1]:
            nid = nxt
            nxt += 1
            out.pos[nid] = q
            out.rad[nid] = r
            out.typ[nid] = mode_type
            chain.append(nid)
        chain.append(remap[end])
        for a, b in zip(chain, chain[1:]):
            out.par[b] = a
        n_fit += 1

    for r in skel.roots():
        if r in out.par:
            out.par[r] = -1
    if verbose:
        print(f"  smoothed {n_fit} paths, passed through {n_raw} short paths")
    return out


# ------------------------------------------------------- mesh validation

def load_dae_mesh(path, recentre_to_volume=True):
    """Return (vertices, triangles) from a Viking COLLADA export, in volume
    coordinates if recentre_to_volume (applies the node-level <translate>)."""
    import re
    data = open(path, 'r', errors='replace').read()
    V = np.fromstring(
        re.search(r'positions-array" count="\d+">(.*?)</float_array>', data, re.S).group(1),
        sep=' ').reshape(-1, 3)
    T = np.fromstring(
        re.search(r'<triangles count="\d+".*?<p>(.*?)</p>', data, re.S).group(1),
        sep=' ', dtype=np.int64).reshape(-1, 3)
    if recentre_to_volume:
        m = re.search(r'<translate>([^<]+)</translate>', data)
        if m:
            V = V + np.fromstring(m.group(1), sep=' ')
    return V, T


def section_planes(verts, tol=0.004):
    """Recover the real section planes from mesh Z values.

    Do not assume a regular grid - sampling on a guessed grid misses the
    planes entirely and silently returns a garbage radius.
    """
    z = np.unique(np.round(verts[:, 2], 4))
    planes, cur = [], [z[0]]
    for v in z[1:]:
        if v - cur[-1] < tol:
            cur.append(v)
        else:
            planes.append(float(np.mean(cur)))
            cur = [v]
    planes.append(float(np.mean(cur)))
    return np.array(planes)


def contour_radii(verts, planes, zmin, zmax, cluster=0.7, tol=0.004):
    """Mean radius of every contour on every section plane in a Z band."""
    out = []
    for p in planes[(planes > zmin) & (planes < zmax)]:
        sl = verts[np.abs(verts[:, 2] - p) < tol][:, :2]
        if len(sl) < 6:
            continue
        lab = fcluster(linkage(sl, 'single'), t=cluster, criterion='distance')
        for L in np.unique(lab):
            q = sl[lab == L]
            if len(q) < 5:
                continue
            r = float(np.linalg.norm(q - q.mean(0), axis=1).mean())
            if r > 0.02:
                out.append(r)
    return np.array(out)


def validate_against_mesh(skel, verts, tris, zmin, zmax, cluster=0.7):
    """Bracket the true centreline length from membrane area: L = A / (2*pi*r).

    IMPORTANT: this is a bracket, not a measurement. The answer depends
    strongly on how the radius is averaged, and for processes that run within
    the section plane rather than across it the contour radius overestimates
    the true tube radius. Treat a skeleton length inside the bracket as
    "not obviously wrong", and calibrate smoothing against hand-traced
    reference branches instead of against this number.
    """
    A, B, C = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    zc = (A[:, 2] + B[:, 2] + C[:, 2]) / 3
    band = (zc > zmin) & (zc < zmax)
    total_area = float(area[band].sum())

    planes = section_planes(verts)
    rs = contour_radii(verts, planes, zmin, zmax, cluster=cluster)
    if rs.size == 0:
        return None
    w = 2 * np.pi * rs
    estimators = {
        'median r': float(np.median(rs)),
        'mean r': float(rs.mean()),
        'area-weighted r': float((rs * w).sum() / w.sum()),
    }
    lengths = {k: total_area / (2 * np.pi * v) for k, v in estimators.items()}

    sk_len = 0.0
    for i, p in skel.par.items():
        if p == -1:
            continue
        zm = (skel.pos[i][2] + skel.pos[p][2]) / 2
        if zmin < zm < zmax:
            sk_len += float(np.linalg.norm(skel.pos[i] - skel.pos[p]))

    lo, hi = min(lengths.values()), max(lengths.values())
    return dict(membrane_area=total_area, n_contours=int(rs.size),
                radii=estimators, lengths=lengths,
                bracket=(lo, hi), skeleton_length=sk_len,
                inside=lo <= sk_len <= hi)


def branch_length_from_tip(skel, tip):
    """Length from a tip back up to the nearest branch point or root."""
    ch = skel.children()
    L, cur = 0.0, tip
    while True:
        p = skel.par.get(cur, -1)
        if p == -1:
            return L
        L += float(np.linalg.norm(skel.pos[cur] - skel.pos[p]))
        if len(ch[p]) != 1:
            return L
        cur = p


def calibrate_smoothing(skel, reference, coefs=np.arange(0.2, 6.01, 0.2),
                        step=0.25, verbose=True):
    """Find the smooth_coef that reproduces hand-measured branch lengths.

    reference: {tip_node_id: measured_length_um} for a handful of terminal
    branches you traced by eye in Blender. Tip ids survive smoothing because
    branch points and tips are held fixed, so they are a stable key.

    Trace 3-5 branches by hand, spread across both compartments and across
    calibres, and pick the coefficient that minimises the error. This is the
    only place manual work is genuinely needed - once the coefficient is set
    it applies to the whole cell, and to other cells from the same volume.

    Returns (best_coef, {coef: mean_abs_pct_error}).
    """
    table = {}
    for coef in coefs:
        test = smooth_resample(skel.copy(), step=step, smooth_coef=float(coef),
                               verbose=False)
        errs = []
        for tip, target in reference.items():
            if tip not in test.pos or target <= 0:
                continue
            L = branch_length_from_tip(test, tip)
            errs.append(abs(L - target) / target)
        if errs:
            table[round(float(coef), 3)] = float(np.mean(errs))
    if not table:
        return None, {}
    best = min(table, key=lambda k: table[k])
    if verbose:
        print(f"  best smooth_coef = {best}  "
              f"(mean abs error {100*table[best]:.1f}% over "
              f"{len(reference)} reference branches)")
    return best, table
