#!/usr/bin/env python3
"""
Synapse density per branch length, from a repaired SWC plus a synapse CSV.

Single cell:
    python density.py repaired.swc synapses.csv --out-dir out/192 --group-by partner

Many cells: use pipeline.py, which calls analyse() below.

CSV columns are auto-detected from common Viking / SBFSEM-pytools names, or
name them explicitly with --x-col / --y-col / --z-col / --type-col /
--partner-col. Everything is in the SWC's coordinate system and units.
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import swcrepair as sr


CANDIDATES = {
    'x': ['x', 'xum', 'volumex', 'xvolume', 'posx', 'centroidx', 'synapsex',
          'synapsexum', 'locationx'],
    'y': ['y', 'yum', 'volumey', 'yvolume', 'posy', 'centroidy', 'synapsey',
          'synapseyum', 'locationy'],
    'z': ['z', 'zum', 'volumez', 'zvolume', 'posz', 'centroidz', 'synapsez',
          'synapsezum', 'locationz', 'section'],
    'type': ['synapsetype', 'structuretype', 'type', 'label', 'tag', 'name'],
    'partner': ['neuronlabel', 'partnerlabel', 'partner', 'partnercell',
                'partnerid', 'neuronid', 'targetcell', 'partnerstructure',
                'targetid'],
    'id': ['synapseid', 'structureid', 'id', 'childid'],
}

# How synapse types map to contact classes. A gap junction and a postsynaptic
# density are different kinds of contact and must not be pooled: one is
# electrical coupling, the other chemical input. Substring matching, first match
# wins, anything unmatched becomes "other".
CONTACT_CLASSES = [
    ("gap_junction", ["GapJunction"]),
    ("psd",          ["RibbonPost", "ConvPost", "Post"]),
    ("presynaptic",  ["ConvPre", "CisternPre", "PlaqueLikePre", "Pre"]),
]
CONTACT_COL = "contact_class"

# Partner label aliases: alias -> canonical. CBbwf is a wide-field CBb, not a
# separate partner class, so counting it separately splits one population in
# two and leaves a one-contact series in every figure.
PARTNER_ALIASES = {"CBbwf": "CBb"}


def normalize_label(value, aliases=None):
    """Fold a partner label onto its canonical form.

    Applied to the partner column in place on load, so the summary, the
    grouping and the figures all see the same set of classes.
    """
    v = (value or "").strip()
    return (aliases if aliases is not None else PARTNER_ALIASES).get(v, v)


def classify_contact(type_value, classes=None):
    t = (type_value or "").strip()
    for name, pats in (classes or CONTACT_CLASSES):
        if any(p in t for p in pats):
            return name
    return "other"


# Fallback for coordinate columns: a name whose alphanumeric form both starts
# with something and ends with the axis letter or 'axis+um'. Covers
# SynapseX_um, X (um), Centroid_Z and similar without matching NeuronID.
COORD_SUFFIXES = {'x': ('x', 'xum'), 'y': ('y', 'yum'), 'z': ('z', 'zum')}


def _norm(s):
    return ''.join(c for c in s.lower() if c.isalnum())


def detect_column(fieldnames, want, override=None):
    if override:
        if override not in fieldnames:
            raise KeyError(f"column '{override}' not in CSV: {fieldnames}")
        return override
    table = {_norm(f): f for f in fieldnames}
    for cand in CANDIDATES[want]:
        if cand in table:
            return table[cand]
    if want in COORD_SUFFIXES:
        for suffix in COORD_SUFFIXES[want]:
            for k, f in table.items():
                if k.endswith(suffix):
                    return f
    return None


def census(path, max_values=20):
    """Report every column and, for the categorical ones, its value counts.

    Run this first on a new export: it is the fastest way to see which columns
    are worth grouping or filtering on, and which rows are not synapses at all.
    """
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    out = {}
    for col in rows[0]:
        vals = [(r.get(col) or '').strip() for r in rows]
        uniq = collections.Counter(vals)
        numeric = 0
        for v in vals[:200]:
            try:
                float(v)
                numeric += 1
            except ValueError:
                pass
        out[col] = dict(n_unique=len(uniq),
                        numeric=numeric > 0.9 * min(len(vals), 200),
                        counts=uniq)
    return out


def print_census(path, max_values=20):
    info = census(path)
    print(f"{path}")
    with open(path, newline='') as fh:
        n = sum(1 for _ in csv.DictReader(fh))
    print(f"{n} rows\n")
    for col, d in info.items():
        kind = 'numeric' if d['numeric'] else f"{d['n_unique']} values"
        print(f"{col}  ({kind})")
        if d['numeric']:
            continue
        for v, c in d['counts'].most_common(max_values):
            print(f"    {(v or '(blank)'):<24} {c}")
        if d['n_unique'] > max_values:
            print(f"    ... and {d['n_unique']-max_values} more")
        print()
    return info


def _parse_selector(spec):
    """'Direction=post' or 'SynapseType=Desmosome,Caveola' -> (col, {values})."""
    if '=' not in spec:
        raise ValueError(f"bad selector {spec!r}, want COLUMN=VALUE[,VALUE...]")
    col, vals = spec.split('=', 1)
    return col.strip(), {v.strip() for v in vals.split(',') if v.strip()}


def load_synapses(path, columns=None, include=(), exclude=(),
                  bidirectional=(), contact_classes=None,
                  partner_aliases=None, log=print):
    """Read a synapse CSV.

    columns        optionally overrides autodetection, e.g. {'x': 'SynapseX_um'}
    include        [(column, {allowed values})] - keep only matching rows
    exclude        [(column, {rejected values})] - drop matching rows
    bidirectional  substrings of the type column whose rows BYPASS `include`

    bidirectional exists for gap junctions. A gap junction is bidirectional, so
    which side Viking's annotator marked is a convention, not biology: on cell
    192, 110 distinct gap junctions split 77 post / 33 pre, all with unique
    structure ids and no duplicated pairs. A blanket Direction=post filter would
    silently discard 30% of them. Exempt rows still respect `exclude`, so
    non-synaptic structures are still dropped.

    Every original column is kept on each record, so any of them can be used
    as a grouping axis later.
    """
    columns = columns or {}
    with open(path, newline='') as fh:
        rdr = csv.DictReader(fh)
        if not rdr.fieldnames:
            raise ValueError(f"{path} has no header row")
        fields = list(rdr.fieldnames)
        if CONTACT_COL not in fields:
            fields.append(CONTACT_COL)   # derived, so it can be grouped on
        cols = {k: detect_column(fields, k, columns.get(k)) for k in CANDIDATES}
        for k in ('x', 'y', 'z'):
            if cols[k] is None:
                raise ValueError(
                    f"could not find the {k} column in {path}; header is "
                    f"{fields}. Name it with --{k}-col.")
        log("  columns: " + ", ".join(f"{k}={v}" for k, v in cols.items() if v))
        for col, _ in list(include) + list(exclude):
            if col not in fields:
                raise ValueError(f"filter column {col!r} not in {path}; "
                                 f"header is {fields}")

        out, skipped, dropped = [], 0, collections.Counter()
        n_exempt = 0
        tcol = cols.get('type')
        for i, row in enumerate(rdr):
            keep = True
            rtype = (row.get(tcol) or '').strip() if tcol else ''
            exempt = bool(bidirectional) and any(b in rtype
                                                 for b in bidirectional)
            if exempt:
                n_exempt += 1
            for col, vals in (() if exempt else include):
                if (row.get(col) or '').strip() not in vals:
                    dropped[f"not {col}={'/'.join(sorted(vals))}"] += 1
                    keep = False
                    break
            if keep:
                for col, vals in exclude:
                    v = (row.get(col) or '').strip()
                    if v in vals:
                        dropped[f"{col}={v}"] += 1
                        keep = False
                        break
            if not keep:
                continue
            try:
                p = np.array([float(row[cols['x']]), float(row[cols['y']]),
                              float(row[cols['z']])])
            except (TypeError, ValueError):
                skipped += 1
                continue
            row[CONTACT_COL] = classify_contact(rtype, contact_classes)
            if cols.get('partner'):
                row[cols['partner']] = normalize_label(
                    row.get(cols['partner']), partner_aliases)
            out.append(dict(
                idx=i,
                syn_id=(row.get(cols['id']) or '').strip() if cols['id'] else str(i),
                pos=p,
                type=(row.get(cols['type']) or 'unspecified').strip()
                     if cols['type'] else 'unspecified',
                partner=(row.get(cols['partner']) or '').strip()
                        if cols['partner'] else '',
                row=row,
            ))
    if n_exempt:
        log(f"  {n_exempt} row(s) matched {list(bidirectional)} and bypassed "
            f"the include filters - treated as bidirectional")
    if dropped:
        log(f"  filtered out {sum(dropped.values())} row(s):")
        for reason, n in dropped.most_common():
            log(f"    {reason:<40} {n}")
    if skipped:
        log(f"  skipped {skipped} row(s) with unparseable coordinates")
    folded = (partner_aliases if partner_aliases is not None
              else PARTNER_ALIASES)
    if folded:
        log("  partner labels folded: "
            + ", ".join(f"{a} -> {b}" for a, b in folded.items()))
    if out:
        tally = collections.Counter(r['row'][CONTACT_COL] for r in out)
        log("  contact classes: "
            + ", ".join(f"{k}={v}" for k, v in tally.most_common()))
    return out, cols, fields


def clip_outside_sphere(P, centre, radius):
    """Keep only the parts of a polyline that lie outside a sphere.

    Returns a list of sub-polylines. Used to take the soma out of the path
    length: collapsing the soma to one node removes the soma chain, but the
    spokes from that node out to each primary dendrite remain, and each is
    about one soma radius long. On cell 192 that is 17.5 um, 2.5% of the total,
    concentrated entirely in the proximal compartment where it would inflate
    the denominator most.
    """
    if radius <= 0:
        return [P]
    d = np.linalg.norm(P - centre, axis=1)
    out, cur = [], []
    for k in range(len(P)):
        if d[k] > radius:
            cur.append(P[k])
        if k + 1 < len(P):
            a, b, da, db = P[k], P[k + 1], d[k], d[k + 1]
            if (da > radius) != (db > radius):
                # linear crossing of the sphere boundary along this segment
                t = (radius - da) / (db - da) if abs(db - da) > 1e-12 else 0.0
                cross = a + np.clip(t, 0.0, 1.0) * (b - a)
                cur.append(cross)
                if da > radius:          # leaving the outside, close this run
                    if len(cur) >= 2:
                        out.append(np.array(cur))
                    cur = []
                else:                    # entering the outside, start fresh
                    cur = [cross]
    if len(cur) >= 2:
        out.append(np.array(cur))
    return out


def branch_table(skel, z_split=0.0, soma_centre=None, soma_radius=0.0,
                 use_compartments=True):
    """One record per unbranched branch, with cumulative arc length.

    Any part of a branch inside the soma sphere is excluded, so the denominator
    is dendritic path length rather than dendritic path length plus a set of
    radial spokes through the cell body.
    """
    depth = skel.depths()
    branches = []
    excluded = 0.0
    dropped = 0
    for path in skel.paths():
        P = np.array([skel.pos[n] for n in path])
        full = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        pieces = (clip_outside_sphere(P, soma_centre, soma_radius)
                  if soma_radius > 0 and soma_centre is not None else [P])
        kept = sum(float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
                   for q in pieces)
        excluded += full - kept
        if not pieces:
            dropped += 1
            continue
        for q in pieces:
            d = np.linalg.norm(np.diff(q, axis=0), axis=1)
            arc = np.concatenate([[0.0], np.cumsum(d)])
            if arc[-1] <= 0:
                continue
            zmean = float(q[:, 2].mean())
            if use_compartments and z_split:
                comp = 'arboreal' if zmean >= z_split else 'lobular'
            else:
                comp = 'cell'
            branches.append(dict(
                branch=len(branches), nodes=path, pts=q, arc=arc,
                length=float(arc[-1]),
                radius=float(np.median([skel.rad[n] for n in path])),
                z_mean=zmean, compartment=comp,
                soma_distance=float(depth.get(path[0], np.nan)),
            ))
    return branches, dict(excluded_length=excluded, dropped_branches=dropped)


def assign(branches, syns, max_dist, sample_step=0.1):
    """Nearest-centreline assignment. The residual distance is reported per
    synapse so ambiguous cases are visible rather than silently trusted."""
    samples, owner = [], []
    for b in branches:
        for k in range(len(b['pts']) - 1):
            a, c = b['pts'][k], b['pts'][k + 1]
            seg = float(np.linalg.norm(c - a))
            n = max(1, int(seg / sample_step))
            for t in np.linspace(0, 1, n, endpoint=False):
                samples.append(a + t * (c - a))
                owner.append((b['branch'], b['arc'][k] + t * seg))
        samples.append(b['pts'][-1])
        owner.append((b['branch'], b['arc'][-1]))
    if not samples:
        return [], [dict(s, distance=float('inf')) for s in syns]
    tree = cKDTree(np.array(samples))

    assigned, orphan = [], []
    for s in syns:
        dist, k = tree.query(s['pos'])
        if dist > max_dist:
            orphan.append(dict(s, distance=float(dist)))
            continue
        bi, arc = owner[int(k)]
        assigned.append(dict(s, branch=bi, arc=float(arc), distance=float(dist)))
    return assigned, orphan


def make_key(group_by, cols):
    """Build a row -> category function.

    group_by is a comma-separated list of real CSV column names, crossed
    together. 'type' and 'partner' are accepted as aliases for whichever
    columns autodetection picked, and 'both' means type crossed with partner.
    """
    spec = [t.strip() for t in str(group_by).split(',') if t.strip()]
    if spec == ['both']:
        spec = ['type', 'partner']
    if spec == ['contact']:
        spec = [CONTACT_COL]
    resolved = []
    for t in spec:
        resolved.append(cols.get(t) if t in ('type', 'partner') and cols.get(t)
                        else t)
    resolved = [r for r in resolved if r]
    if not resolved:
        resolved = [cols.get('type') or 'type']

    def key(a):
        parts = []
        for col in resolved:
            v = (a['row'].get(col) if a.get('row') else None)
            if v is None:
                v = a.get(col, '')
            v = (v or '').strip() or 'unlabeled'
            parts.append(v)
        return '|'.join(parts)

    return key, resolved


def analyse(swc_path, csv_path, out_dir, z_split=0.0, max_dist=3.0,
            use_compartments=True,
            group_by='type', exclude_soma_radius=0.0, columns=None,
            include=(), exclude=(), bidirectional=(), contact_classes=None,
            partner_aliases=None, cell_id=None, volume='', log=print):
    """Run the whole density analysis. Returns a result dict and writes
    branches.csv, synapses.csv and summary.csv into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skel = sr.Skel.load(swc_path)
    bad = skel.check()
    if bad:
        raise ValueError(f"{swc_path} has {len(bad)} structural errors; "
                         f"repair it first: {bad[:2]}")
    log(f"  skeleton {len(skel.pos)} nodes, {skel.total_length():.1f} um, "
        f"{len(skel.roots())} root(s)")

    syns, cols, fields = load_synapses(csv_path, columns, include=include,
                                       exclude=exclude,
                                       bidirectional=bidirectional,
                                       contact_classes=contact_classes,
                                       partner_aliases=partner_aliases,
                                       log=log)
    log(f"  synapses {len(syns)} rows kept")
    if not syns:
        raise ValueError(f"no rows left in {csv_path} after filtering")

    if use_compartments and not z_split:
        raise ValueError(
            "use_compartments is on but z_split is unset. Set "
            "z_split_section in cells.csv, or turn compartments off.")

    soma_centre = soma_radius = None
    if skel.roots():
        root = min(skel.roots())
        soma_centre = skel.pos[root]
        if exclude_soma_radius < 0:
            soma_radius = float(skel.rad[root])
            log(f"  soma exclusion: auto, {soma_radius:.3f} um "
                f"(the soma node's own radius)")
        else:
            soma_radius = float(exclude_soma_radius)
            if soma_radius > 0:
                log(f"  soma exclusion: {soma_radius:.3f} um as set")
            else:
                log(f"  soma exclusion: OFF - the spokes from the soma centre "
                    f"out to each primary dendrite stay in the denominator")

    branches, bstat = branch_table(skel, z_split, soma_centre,
                                   soma_radius or 0,
                                   use_compartments=use_compartments)
    if use_compartments:
        n_lob = sum(1 for b in branches if b['compartment'] == 'lobular')
        log(f"  branches {len(branches)} (lobular {n_lob}, "
            f"arboreal {len(branches)-n_lob})")
    else:
        log(f"  branches {len(branches)}, whole cell "
            f"(compartments off - set use_compartments = true to split them)")
    if bstat['excluded_length'] > 0:
        total_kept = sum(b['length'] for b in branches)
        log(f"  excluded {bstat['excluded_length']:.2f} um of path inside the "
            f"soma ({100*bstat['excluded_length']/max(total_kept+bstat['excluded_length'],1e-9):.1f}%"
            f" of the skeleton)")
    if bstat['dropped_branches']:
        log(f"  {bstat['dropped_branches']} branch(es) lay entirely inside the "
            f"soma and were dropped")

    if soma_radius and soma_radius > 0:
        before = len(syns)
        syns = [s for s in syns
                if np.linalg.norm(s['pos'] - soma_centre) > soma_radius]
        if before - len(syns):
            log(f"  dropped {before-len(syns)} synapse(s) inside the soma - "
                f"the same boundary as the length exclusion, so numerator and "
                f"denominator agree")

    assigned, orphan = assign(branches, syns, max_dist)
    frac_orphan = len(orphan) / max(1, len(syns))
    log(f"  assigned {len(assigned)}, unassigned {len(orphan)} "
        f"({100*frac_orphan:.1f}%)")
    warnings = []
    if orphan:
        d = sorted(o['distance'] for o in orphan)
        log(f"    unassigned distance to centreline: min {d[0]:.2f}, "
            f"median {d[len(d)//2]:.2f}, max {d[-1]:.2f} um")
        if frac_orphan > 0.05:
            w = (f"{100*frac_orphan:.1f}% of synapses unassigned - max_dist too "
                 f"small, skeleton not reaching distal tips, or the two files "
                 f"are not in the same coordinate frame")
            warnings.append(w)
            log(f"    WARNING: {w}")
    residual_median = residual_p95 = float('nan')
    if assigned:
        res = np.array([a['distance'] for a in assigned])
        residual_median = float(np.median(res))
        residual_p95 = float(np.percentile(res, 95))
        log(f"    residual distance to centreline: median {residual_median:.3f}, "
            f"p95 {residual_p95:.3f} um")

    key, group_cols = make_key(group_by, cols)
    for col in group_cols:
        if col not in fields:
            raise ValueError(f"group-by column {col!r} not in {csv_path}; "
                             f"header is {fields}")
    log(f"  grouping by {' x '.join(group_cols)}")

    by_branch = collections.defaultdict(collections.Counter)
    for a in assigned:
        by_branch[a['branch']][key(a)] += 1
    cats = sorted({key(a) for a in assigned})

    branches_csv = out_dir / 'branches.csv'
    with open(branches_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['volume', 'cell_id', 'branch', 'compartment', 'length_um',
                    'median_radius_um', 'z_mean', 'soma_path_distance_um',
                    'n_total', 'density_per_um']
                   + [f'n_{c}' for c in cats] + [f'density_{c}' for c in cats])
        for b in branches:
            c = by_branch[b['branch']]
            n = sum(c.values())
            L = b['length']
            w.writerow([volume, cell_id or '', b['branch'], b['compartment'],
                        round(L, 4), round(b['radius'], 4),
                        round(b['z_mean'], 3), round(b['soma_distance'], 3),
                        n, round(n / L, 5) if L > 0 else '']
                       + [c[k] for k in cats]
                       + [round(c[k] / L, 5) if L > 0 else '' for k in cats])

    synapses_csv = out_dir / 'synapses.csv'
    bmap = {b['branch']: b for b in branches}
    with open(synapses_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['volume', 'cell_id', 'synapse_id', 'row', 'category', 'type',
                    'partner', 'branch', 'compartment', 'arc_um',
                    'soma_path_distance_um', 'residual_um', 'x', 'y', 'z']
                   + fields)
        for a in assigned:
            b = bmap[a['branch']]
            w.writerow([volume, cell_id or '', a['syn_id'], a['idx'], key(a), a['type'],
                        a['partner'], a['branch'], b['compartment'],
                        round(a['arc'], 4),
                        round(b['soma_distance'] + a['arc'], 4),
                        round(a['distance'], 4)]
                       + [round(v, 4) for v in a['pos']]
                       + [a['row'].get(f, '') for f in fields])
        for o in orphan:
            w.writerow([volume, cell_id or '', o['syn_id'], o['idx'], key(o), o['type'],
                        o['partner'], '', 'UNASSIGNED', '', '',
                        round(o['distance'], 4)]
                       + [round(v, 4) for v in o['pos']]
                       + [o['row'].get(f, '') for f in fields])

    # contact class per synapse, so the summary keeps gap junctions and
    # postsynaptic densities apart. Pooling them would be wrong: one is
    # electrical coupling, the other chemical input, and on cell 192 the same
    # partner class carries both (CBb: 44 gap junctions and 7 ribbons).
    by_class = collections.defaultdict(lambda: collections.defaultdict(int))
    for a in assigned:
        b = bmap[a['branch']]
        by_class[(b['compartment'], a['row'][CONTACT_COL])][key(a)] += 1

    comps = (('lobular', 'arboreal') if use_compartments else ('cell',))
    rows = []
    for comp in comps:
        L = sum(b['length'] for b in branches if b['compartment'] == comp)
        nb = sum(1 for b in branches if b['compartment'] == comp)
        classes = sorted({c for (cp, c) in by_class if cp == comp})
        for cl in classes:
            sub = by_class[(comp, cl)]
            for c in sorted(sub):
                rows.append([volume, cell_id or '', comp, cl, c, sub[c], nb,
                             round(L, 3),
                             round(sub[c] / L, 5) if L > 0 else ''])
            tot = sum(sub.values())
            rows.append([volume, cell_id or '', comp, cl, 'ALL', tot, nb,
                         round(L, 3), round(tot / L, 5) if L > 0 else ''])
        n_all = sum(sum(by_branch[b['branch']].values()) for b in branches
                    if b['compartment'] == comp)
        rows.append([volume, cell_id or '', comp, 'ALL', 'ALL', n_all, nb,
                     round(L, 3), round(n_all / L, 5) if L > 0 else ''])

    summary_csv = out_dir / 'summary.csv'
    with open(summary_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['volume', 'cell_id', 'compartment', 'contact_class',
                    'category', 'n', 'n_branches', 'length_um',
                    'density_per_um'])
        w.writerows(rows)

    return dict(
        cell_id=cell_id, swc=str(swc_path), csv=str(csv_path),
        skeleton_length=skel.total_length(), n_branches=len(branches),
        n_synapses=len(syns), n_assigned=len(assigned), n_unassigned=len(orphan),
        excluded_soma_length=bstat['excluded_length'],
        soma_radius=soma_radius,
        frac_unassigned=frac_orphan, residual_median=residual_median,
        residual_p95=residual_p95, categories=cats, z_split=z_split,
        group_by=group_by, group_cols=group_cols, rows=rows, warnings=warnings,
        branches_csv=str(branches_csv), synapses_csv=str(synapses_csv),
        summary_csv=str(summary_csv),
    )


def print_summary(res):
    cols = res.get('group_cols') or [res['group_by']]
    print(f"\ncompartment x contact class x {' x '.join(cols)}")
    print(f"{'compartment':<10} {'contact':<13} {'category':<22} {'n':>5} "
          f"{'length_um':>10} {'per_um':>9}")
    last = None
    for r in res['rows']:
        if last is not None and (r[2], r[3]) != last:
            print()
        last = (r[2], r[3])
        print(f"{r[2]:<10} {r[3]:<13} {r[4]:<22} {r[5]:>5} {r[7]:>10} "
              f"{str(r[8]):>9}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('swc')
    p.add_argument('csv')
    p.add_argument('--out-dir', default='.', help='directory for the three CSVs')
    p.add_argument('--volume', default='',
                   help='written into a volume column for provenance')
    p.add_argument('--cell-id', default=None,
                   help='written into a cell_id column so outputs concatenate')
    p.add_argument('--no-compartments', dest='use_compartments',
                   action='store_false',
                   help='pool the whole cell instead of splitting it at '
                        '--z-split. Not recommended for density: the '
                        'denominator then includes dendrite that could not '
                        'have received the input.')
    p.add_argument('--z-split', type=float, default=0.0,
                   help='compartment boundary in Z, volume coordinates')
    p.add_argument('--max-dist', type=float, default=3.0)
    p.add_argument('--group-by', default='type',
                   help="comma-separated CSV column names to cross, e.g. "
                        "'SynapseType' or 'SynapseType,NeuronLabel'. 'type' and "
                        "'partner' are aliases for the autodetected columns.")
    p.add_argument('--filter', action='append', default=[], metavar='COL=VAL[,VAL]',
                   help='keep only rows matching. Repeatable. e.g. Direction=post')
    p.add_argument('--exclude-rows', action='append', default=[],
                   metavar='COL=VAL[,VAL]',
                   help='drop rows matching. Repeatable.')
    p.add_argument('--bidirectional', action='append', default=['GapJunction'],
                   metavar='SUBSTRING',
                   help='synapse types that bypass --filter, because they are '
                        'bidirectional. Defaults to GapJunction.')
    p.add_argument('--alias', action='append', default=[],
                   metavar='FROM=TO',
                   help='fold a partner label onto another, e.g. CBbwf=CBb. '
                        'Repeatable. Defaults to CBbwf=CBb.')
    p.add_argument('--census', action='store_true',
                   help='print the CSV column census and exit. Run this first '
                        'on a new export.')
    p.add_argument('--exclude-soma-radius', type=float, default=-1.0,
                   help='radius in um around the soma to exclude from BOTH the '
                        'path length and the synapse count. -1 (default) uses '
                        'the soma node\'s own radius; 0 disables it.')
    for k in CANDIDATES:
        p.add_argument(f'--{k}-col', default=None)
    args = p.parse_args()

    if args.census:
        print_census(args.csv)
        return
    if args.use_compartments and not args.z_split:
        sys.exit("--z-split is required. Pass --no-compartments to pool the "
                 "whole cell instead, but see the help for why that deflates "
                 "the density.")

    cols = {k: getattr(args, f'{k}_col', None) for k in CANDIDATES}
    try:
        aliases = dict(PARTNER_ALIASES)
        for spec in args.alias:
            if '=' not in spec:
                sys.exit(f"bad --alias {spec!r}, want FROM=TO")
            a, b = spec.split('=', 1)
            aliases[a.strip()] = b.strip()
        include = [_parse_selector(s) for s in args.filter]
        exclude = [_parse_selector(s) for s in args.exclude_rows]
        res = analyse(args.swc, args.csv, args.out_dir, z_split=args.z_split,
                      use_compartments=args.use_compartments,
                      max_dist=args.max_dist, group_by=args.group_by,
                      exclude_soma_radius=args.exclude_soma_radius,
                      columns=cols, include=include, exclude=exclude,
                      bidirectional=tuple(args.bidirectional),
                      partner_aliases=aliases,
                      cell_id=args.cell_id, volume=args.volume)
    except (ValueError, KeyError) as e:
        sys.exit(f"error: {e}")
    print_summary(res)
    print(f"\nwrote {res['branches_csv']}")
    print(f"      {res['synapses_csv']}")
    print(f"      {res['summary_csv']}")


if __name__ == '__main__':
    main()
