#!/usr/bin/env python3
"""
Two-stage SWC repair. Functions take explicit paths so pipeline.py can drive
them across many cells; the CLI below handles one cell at a time.

  python repair_run.py audit  in/c192.swc --out-dir out/192
  # review out/192/regions.csv, set each action to merge or keep
  python repair_run.py repair in/c192.swc --out-dir out/192 \
         --mesh in/Morphology-192.dae
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import swcrepair as sr


REGION_FIELDS = ['region', 'kind', 'action', 'n_nodes', 'length_involved',
                 'n_keep', 'n_drop', 'drop_length', 'tortuosity',
                 'deviation', 'mesh_distance', 'gap', 'fragment_size',
                 'mean_radius',
                 'x', 'y', 'z', 'separate_components',
                 'keep_ids', 'drop_ids', 'bad_ids', 'node_ids']


def _stats_line(skel, log):
    s = skel.stats()
    log(f"  nodes {s['nodes']}  roots {s['roots']}  tips {s['tips']}  "
        f"bifurcations {s['bifurcations']}  multifurcations {s['multifurcations']}")
    log(f"  path length {s['length']:.1f} um")
    bad = skel.check()
    if bad:
        log(f"  structural errors: {len(bad)}  {bad[:2]}")
    return s


def read_manifest(path):
    """Rows as [{region, action, nodes:set}]. Missing file means nothing reviewed.

    Region numbers are NOT stable: they come from clustering, and repair
    re-detects on a skeleton that has already had marker nodes stripped and the
    soma collapsed, so the numbering shifts. Node ids are stable, because
    deletion never renumbers. So decisions are matched back by node overlap and
    the region number is only a label for you to read.
    """
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            try:
                rid = int(row['region'])
            except (KeyError, TypeError, ValueError):
                continue
            nodes = set()
            for tok in (row.get('node_ids') or '').split():
                try:
                    nodes.add(int(tok))
                except ValueError:
                    pass
            out.append(dict(region=rid,
                            kind=(row.get('kind') or 'duplicate').strip().lower(),
                            action=(row.get('action') or '').strip().lower(),
                            nodes=nodes))
    return out


def match_manifest(region, manifest, min_overlap=0.34):
    """Find the reviewed row describing this detected region, by node overlap."""
    nodes = set(region['nodes'])
    if not nodes:
        return None
    best, best_score = None, 0.0
    for row in manifest:
        if not row['nodes']:
            continue
        inter = len(nodes & row['nodes'])
        if not inter:
            continue
        score = inter / len(nodes | row['nodes'])
        if score > best_score:
            best, best_score = row, score
    return best if best_score >= min_overlap else None


def write_manifest(path, regions, preserve=True, skel=None):
    """Write the review manifest, keeping any actions already decided.

    Prior decisions are carried over by node-id overlap, not by region number,
    for the same reason repair matches that way: the numbering comes from
    clustering and can shift between runs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = read_manifest(path) if preserve else []
    n_review = 0
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(REGION_FIELDS)
        for d in regions:
            match = match_manifest(d, prior) if prior else None
            action = (match['action'] if match and match['action'] else 'review')
            if action == 'review':
                n_review += 1
            kind = d.get('kind', 'duplicate')
            plan = None
            if skel is not None and kind == 'duplicate':
                try:
                    plan = sr.merge_plan(skel, d)
                except Exception:
                    plan = None
            keep_ids = ' '.join(str(n) for n in plan['keep']) if plan else ''
            drop_ids = ' '.join(str(n) for n in plan['drop']) if plan else ''
            w.writerow([d['region'], kind, action,
                        len(d['nodes']), round(d['length_involved'], 3),
                        len(plan['keep']) if plan else '',
                        len(plan['drop']) if plan else '',
                        round(plan['drop_length'], 3) if plan else '',
                        round(d['tortuosity'], 2) if d.get('tortuosity') else '',
                        round(d['max_deviation'], 3) if d.get('max_deviation') else '',
                        round(d['max_distance'], 3) if d.get('max_distance') else '',
                        round(d['gap'], 3) if d.get('gap') is not None else '',
                        d.get('fragment_size', ''),
                        round(d['mean_radius'], 4),
                        round(d['centroid'][0], 3), round(d['centroid'][1], 3),
                        round(d['centroid'][2], 3),
                        int(d['separate_components']),
                        keep_ids, drop_ids,
                        ' '.join(str(n) for n in d.get('bad', [])),
                        ' '.join(str(n) for n in d['nodes'])])
    return path, n_review


def audit(swc_path, out_dir, soma_type=1, min_tree_dist=6.0, strip_types=(6,),
          mesh=None, log=print):
    """Structural report plus a region manifest. Never modifies the input."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skel = sr.Skel.load(swc_path)
    log(f"=== audit {swc_path} ===")
    s = _stats_line(skel, log)

    ch = skel.children()

    def subtree_size(r):
        n, st = 0, [r]
        while st:
            x = st.pop()
            n += 1
            st.extend(ch[x])
        return n

    roots = skel.roots()
    components = sorted(((subtree_size(r), r) for r in roots), reverse=True)
    if len(roots) > 1:
        log(f"\ndisconnected components: {len(roots)}")
        for n, r in components[:10]:
            log(f"  root {r:>6}  {n:>5} nodes  Z={skel.pos[r][2]:.2f}  "
                f"r={skel.rad[r]:.3f}")

    markers = {}
    for t in strip_types:
        d = sr.marker_diagnosis(skel, t)
        if d:
            markers[t] = d
            log(f"\ntype {t} marker check: n={d['n']}, "
                f"{100*d['tip_fraction']:.0f}% tips, "
                f"offset/parent_radius={d['offset_over_radius']:.2f} "
                f"(corr {d['correlation']:.2f}), carries "
                f"{d['length_carried']:.1f} um")

    # Detect on a copy with markers stripped and the soma collapsed, so the
    # node sets and the merge plans match exactly what repair will act on.
    staged = skel.copy()
    if strip_types:
        sr.strip_types(staged, tuple(strip_types), verbose=False)
    sr.collapse_soma(staged, soma_type=soma_type, verbose=False)

    regions, fragments = sr.split_fragments(
        sr.find_duplicates(staged, min_tree_dist=min_tree_dist))
    for d in regions:
        d['kind'] = 'duplicate'
    for d in fragments:
        info = sr.describe_fragment(staged, d)
        if info:
            d['gap'] = info['nearest'][1]
            d['fragment_size'] = info['fragment_size']
    found = {n for d in regions + fragments for n in d['nodes']}

    def fresh(cands, kind):
        out = []
        for d in cands:
            if set(d['nodes']) & found:
                continue
            d['kind'] = kind
            out.append(d)
            found.update(d['nodes'])
        return out

    hairpins = fresh(sr.find_hairpins(staged), 'hairpin')
    offmesh = []
    if mesh:
        try:
            V, _ = sr.load_dae_mesh(mesh)
            offmesh = fresh(sr.find_offmesh(staged, V), 'offmesh')
        except Exception as e:
            log(f"\ncould not check against {mesh}: {e}")
    else:
        log(f"\nno mesh supplied - skipping the off-mesh check, which is the "
            f"strongest one available")
    spikes = fresh(sr.find_spikes(staged), 'spike')
    phantom = sum(d['length_involved'] / 2 for d in regions)
    if fragments:
        log(f"\ndisconnected fragments - the skeleton is BROKEN here, not "
            f"doubled. These")
        log(f"satisfy the duplicate test only because an infinite tree "
            f"distance trivially")
        log(f"does. Merging one would delete a chain. Use action=graft to join "
            f"it on, or")
        log(f"drop to delete it.")
        log(f"{'region':>7} {'size':>5} {'gap_um':>7} {'Z':>8}")
        for d in fragments:
            log(f"{d['region']:>7} {d.get('fragment_size',''):>5} "
                f"{d.get('gap',0):>7.2f} {d['centroid'][2]:>8.2f}")

    log(f"\nsuspected duplicated processes: {len(regions)}, "
        f"~{phantom:.1f} um phantom "
        f"({100*phantom/max(s['length'],1e-9):.1f}% of raw length)")
    if regions:
        log(f"{'region':>7} {'nodes':>6} {'len_um':>8} {'mean_r':>7} {'Z':>8} "
            f"{'sep_comp':>9}")
        for d in regions:
            log(f"{d['region']:>7} {len(d['nodes']):>6} "
                f"{d['length_involved']:>8.1f} {d['mean_radius']:>7.3f} "
                f"{d['centroid'][2]:>8.2f} {str(d['separate_components']):>9}")

    if regions:
        log(f"\nwhat each merge would delete (from merge_plan, no-ops omitted):")
        log(f"{'region':>7} {'keep':>5} {'drop':>5} {'keep_um':>8} "
            f"{'drop_um':>8} {'Z':>8}")
        for d in regions:
            plan = sr.merge_plan(staged, d)
            if plan is None:
                log(f"{d['region']:>7}   no-op, would change nothing")
                continue
            log(f"{d['region']:>7} {len(plan['keep']):>5} {len(plan['drop']):>5} "
                f"{plan['keep_length']:>8.2f} {plan['drop_length']:>8.2f} "
                f"{d['centroid'][2]:>8.2f}")

    if hairpins:
        log(f"\nhairpins - a path folding back on itself. Different signature "
            f"from a duplicate,")
        log(f"so these are reported but NOT auto-merged. Fix upstream in "
            f"Viking, or set")
        log(f"action=trim to delete the fold and connect across it.")
        log(f"{'region':>7} {'nodes':>6} {'len_um':>7} {'chord':>7} "
            f"{'tort':>6} {'Z':>8}")
        for d in hairpins:
            log(f"{d['region']:>7} {len(d['nodes']):>6} "
                f"{d['length_involved']:>7.2f} {d['chord']:>7.2f} "
                f"{d['tortuosity']:>6.2f} {d['centroid'][2]:>8.2f}")

    if offmesh:
        log(f"\noff-mesh - nodes sitting outside the membrane they came from. "
            f"The mesh is")
        log(f"ground truth, so these are wrong, not merely suspicious. "
            f"action=snip deletes them.")
        log(f"{'region':>7} {'bad':>4} {'dist_um':>8} {'r':>6} {'Z':>8}")
        for d in offmesh:
            log(f"{d['region']:>7} {d['n_off_nodes']:>4} "
                f"{d['max_distance']:>8.2f} {d['mean_radius']:>6.3f} "
                f"{d['centroid'][2]:>8.2f}")
    if spikes:
        log(f"\nspikes - short excursions off a branch's own course, measured "
            f"against a")
        log(f"moving average of the neighbours. action=snip deletes the "
            f"flagged nodes.")
        log(f"{'region':>7} {'bad':>4} {'maxdev':>8} {'r':>6} {'Z':>8}")
        for d in spikes:
            log(f"{d['region']:>7} {d['n_spike_nodes']:>4} "
                f"{d['max_deviation']:>8.2f} {d['mean_radius']:>6.3f} "
                f"{d['centroid'][2]:>8.2f}")

    manifest, n_review = write_manifest(
        out_dir / 'regions.csv',
        regions + fragments + hairpins + offmesh + spikes, skel=staged)
    log(f"\nwrote {manifest}  ({n_review} region(s) still marked review)")
    return dict(swc=str(swc_path), stats=s, n_components=len(roots),
                components=components, markers=markers, regions=regions,
                fragments=fragments, hairpins=hairpins, offmesh=offmesh,
                spikes=spikes,
                phantom_length=phantom,
                manifest=str(manifest), n_review=n_review)


def rescale_z(skel, z_scale=1.0, z_offset=0.0, log=print):
    """Apply z_um = z_raw * z_scale + z_offset in place.

    Only needed when an SWC reports Z as a section index instead of
    micrometres. Radii are left alone: they come from in-plane contours and are
    already in the same units as X and Y.
    """
    if z_scale == 1.0 and z_offset == 0.0:
        return skel
    for i in skel.pos:
        skel.pos[i][2] = skel.pos[i][2] * z_scale + z_offset
    zs = [skel.pos[i][2] for i in skel.pos]
    log(f"  rescaled Z by *{z_scale} +{z_offset} -> "
        f"{min(zs):.2f} .. {max(zs):.2f}")
    return skel


def _apply_region_fixes(skel, manifest_rows, mesh, min_tree_dist,
                        graft_within, unreviewed_action, log):
    """Apply the reviewed fixes in regions.csv. Returns (skel, counters)."""
    regions, fragments = sr.split_fragments(
        sr.find_duplicates(skel, min_tree_dist=min_tree_dist))
    for d in regions:
        d['kind'] = 'duplicate'
    found = {n for d in regions + fragments for n in d['nodes']}

    def fresh(cands, kind):
        out = []
        for d in cands:
            if set(d['nodes']) & found:
                continue
            d['kind'] = kind
            out.append(d)
            found.update(d['nodes'])
        return out

    hairpins = fresh(sr.find_hairpins(skel), 'hairpin')
    offmesh = []
    if mesh:
        try:
            V, _ = sr.load_dae_mesh(mesh)
            offmesh = fresh(sr.find_offmesh(skel, V), 'offmesh')
        except Exception as e:
            log(f"  off-mesh check skipped: {e}")
    spikes = fresh(sr.find_spikes(skel), 'spike')

    c = dict(merged=0, kept=0, unreviewed=0, rolled_back=0, unmatched=0,
             trimmed=0, snipped=0, grafted_fragments=0, dropped_fragments=0)

    for d in regions + fragments + hairpins + offmesh + spikes:
        row = match_manifest(d, manifest_rows) if manifest_rows else None
        if row is None:
            if manifest_rows:
                c['unmatched'] += 1
                log(f"  region {d['region']} at Z={d['centroid'][2]:.2f} has no "
                    f"reviewed match in the manifest")
            act = 'merge' if not manifest_rows else 'review'
        else:
            act = row['action'] or 'review'
        if act == 'review':
            act = unreviewed_action if unreviewed_action != 'skip' else 'review'
            if act == 'review':
                c['unreviewed'] += 1
                continue
        if act == 'keep':
            c['kept'] += 1
            continue

        kind = d.get('kind')
        snap = skel.copy()
        if kind == 'fragment':
            if act in ('graft', 'merge'):
                if act == 'merge':
                    log(f"  region {d['region']} is a fragment, not a "
                        f"duplicate - treating 'merge' as 'graft'")
                if sr.graft_region(skel, d, max_gap=graft_within):
                    if skel.check():
                        log(f"  region {d['region']}: graft would break the "
                            f"tree - rolled back")
                        skel = snap
                    else:
                        c['grafted_fragments'] += 1
            elif act == 'drop':
                if sr.drop_region(skel, d):
                    if skel.check():
                        skel = snap
                    else:
                        c['dropped_fragments'] += 1
            else:
                log(f"  region {d['region']} is a fragment; action '{act}' "
                    f"does not apply. Use graft, drop, or keep.")
                c['unreviewed'] += 1
        elif kind in ('offmesh', 'spike'):
            if act == 'snip':
                if sr.snip_region(skel, d):
                    if skel.check():
                        log(f"  region {d['region']}: snip would break the "
                            f"tree - rolled back")
                        skel = snap
                    else:
                        c['snipped'] += 1
            else:
                log(f"  region {d['region']} is a {kind}; action '{act}' does "
                    f"not apply. Use snip or keep.")
                c['unreviewed'] += 1
        elif kind == 'hairpin':
            if act == 'trim':
                if sr.trim_region(skel, d):
                    if skel.check():
                        log(f"  region {d['region']}: trim would break the "
                            f"tree - rolled back")
                        skel = snap
                    else:
                        c['trimmed'] += 1
            else:
                log(f"  region {d['region']} is a hairpin; action '{act}' does "
                    f"not apply. Use trim, keep, or fix it in Viking.")
                c['unreviewed'] += 1
        else:
            if sr.merge_region(skel, d):
                if skel.check():
                    log(f"  region {d['region']}: merge would break the tree "
                        f"- rolled back")
                    skel = snap
                    c['rolled_back'] += 1
                else:
                    c['merged'] += 1

    log(f"  merged {c['merged']}, grafted {c['grafted_fragments']}, "
        f"dropped {c['dropped_fragments']}, trimmed {c['trimmed']}, "
        f"snipped {c['snipped']}, kept {c['kept']}, "
        f"unreviewed {c['unreviewed']}, rolled back {c['rolled_back']}"
        + (f", unmatched {c['unmatched']}" if c['unmatched'] else ""))
    if c['unreviewed']:
        log(f"  NOTE: {c['unreviewed']} region(s) left untouched. Set their "
            f"action in the manifest, or pass --merge-unreviewed.")
    return skel, c


def repair(swc_path, out_swc, manifest=None, mesh=None, bands=(),
           soma_type=1, strip_types=(6,), min_tree_dist=6.0, graft_within=2.0,
           drop_smaller_than=1, step=0.25, smooth_coef=0.8,
           z_scale=1.0, z_offset=0.0, unreviewed_action='skip',
           apply_region_fixes=False, log=print):
    """Apply every fix in order. Returns a result dict; writes out_swc.

    apply_region_fixes=False is the default: the detectors are advisory, you
    fix the skeleton by hand in Blender, and this stage only does the
    mechanical work that no judgement is needed for. Set it True to have the
    reviewed actions in regions.csv applied instead.
    """
    out_swc = Path(out_swc)
    out_swc.parent.mkdir(parents=True, exist_ok=True)
    skel = sr.Skel.load(swc_path)
    log(f"=== repair {swc_path} ===")
    rescale_z(skel, z_scale, z_offset, log=log)
    before = _stats_line(skel, log)

    log("\n--- 0. appended marker nodes ---")
    n_stripped = sr.strip_types(skel, tuple(strip_types), verbose=True) \
        if strip_types else 0
    if not strip_types:
        log("  skipped (strip_types empty)")

    log("\n--- 1. soma collapse ---")
    _, n_soma = sr.collapse_soma(skel, soma_type=soma_type)
    if not n_soma:
        log("  nothing to collapse")

    log("\n--- 2. detector fixes ---")
    if apply_region_fixes:
        manifest_rows = read_manifest(manifest) if manifest else []
        skel, counters = _apply_region_fixes(
            skel, manifest_rows, mesh, min_tree_dist, graft_within,
            unreviewed_action, log)
    else:
        counters = dict(merged=0, kept=0, unreviewed=0, rolled_back=0,
                        unmatched=0, trimmed=0, snipped=0,
                        grafted_fragments=0, dropped_fragments=0)
        log("  OFF - the detectors are advisory only.")
        log("  Fix the skeleton by hand in the Blender add-on and export "
            "edited.swc;")
        log("  this stage reads that automatically next time. Set "
            "apply_region_fixes")
        log("  = true in cells.toml if you want regions.csv applied instead.")

    log("\n--- 3. orphan fragments ---")
    grafted, dropped = sr.handle_orphans(
        skel, graft_within=graft_within, drop_smaller_than=drop_smaller_than)
    if not (grafted or dropped):
        log("  none")

    log("\n--- 4. multifurcations ---")
    n_multi = sr.split_multifurcations(skel)
    if not n_multi:
        log("  none")

    log("\n--- 5. smooth and resample ---")
    pre_smooth = skel.total_length()
    skel = sr.smooth_resample(skel, step=step, smooth_coef=smooth_coef)

    log("")
    after = _stats_line(skel, log)
    bad = skel.check()
    if bad:
        log(f"  !! {len(bad)} structural errors remain, not writing output")
        return dict(ok=False, errors=bad, swc=str(swc_path))

    log(f"\nlength budget")
    log(f"  as loaded              {before['length']:>8.1f} um")
    log(f"  after topology fixes   {pre_smooth:>8.1f} um")
    log(f"  after smoothing        {after['length']:>8.1f} um")
    if after['length'] > 0:
        log(f"  correction             "
            f"{100*(before['length']/after['length']-1):>7.1f}%")

    validation = []
    if mesh and bands:
        log(f"\nindependent check against {mesh}")
        V, T = sr.load_dae_mesh(mesh)
        for zmin, zmax, label in bands:
            v = sr.validate_against_mesh(skel, V, T, zmin, zmax)
            if v is None:
                log(f"  {label}: no contours found in Z {zmin}..{zmax}")
                continue
            v['label'] = label
            v['zmin'], v['zmax'] = zmin, zmax
            validation.append(v)
            log(f"  {label} (Z {zmin}..{zmax}): area {v['membrane_area']:.0f} "
                f"um2, {v['n_contours']} contours")
            for k in v['lengths']:
                log(f"    L from {k:<18s} r={v['radii'][k]:.3f} -> "
                    f"{v['lengths'][k]:7.1f} um")
            lo, hi = v['bracket']
            log(f"    bracket  {lo:7.1f} .. {hi:7.1f} um   skeleton "
                f"{v['skeleton_length']:7.1f} um  "
                f"{'INSIDE' if v['inside'] else 'OUTSIDE'}")
        log("  NOTE: the bracket is wide because it depends on how the contour")
        log("  radius is averaged. It catches gross errors only; calibrate")
        log("  smooth_coef against hand-traced branches for a real number.")

    skel.save(out_swc, extra_header=[
        'REPAIRED by swcrepair',
        f'source {Path(swc_path).name}',
        f'stripped types {list(strip_types)}, soma collapsed, '
        f'multifurcations split, resampled at {step} um with '
        f'smooth_coef {smooth_coef}',
        f'detector fixes {"applied" if apply_region_fixes else "not applied"}',
    ])
    log(f"\nwrote {out_swc}")
    res = dict(ok=True, swc=str(swc_path), out=str(out_swc),
               before=before, after=after, pre_smooth_length=pre_smooth,
               n_stripped=n_stripped, n_soma_collapsed=n_soma,
               grafted=grafted, dropped=dropped, n_multifurcations=n_multi,
               validation=validation, smooth_coef=smooth_coef, step=step,
               applied_region_fixes=apply_region_fixes)
    res.update(counters)
    return res


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    a = sub.add_parser('audit')
    a.add_argument('swc')
    a.add_argument('--out-dir', default='.')
    a.add_argument('--soma-type', type=int, default=1)
    a.add_argument('--min-tree-dist', type=float, default=6.0)
    a.add_argument('--strip-types', type=int, nargs='*', default=[6])
    a.add_argument('--mesh', default=None,
                   help='the .dae, enabling the off-mesh check')

    r = sub.add_parser('repair')
    r.add_argument('swc')
    r.add_argument('--out-dir', default='.')
    r.add_argument('--out', default=None, help='defaults to <out-dir>/repaired.swc')
    r.add_argument('--manifest', default=None,
                   help='defaults to <out-dir>/regions.csv if it exists')
    r.add_argument('--mesh', default=None)
    r.add_argument('--band', action='append', default=[], metavar='ZMIN:ZMAX:LABEL',
                   help='validation band, repeatable, e.g. 63:72.5:arboreal')
    r.add_argument('--soma-type', type=int, default=1)
    r.add_argument('--strip-types', type=int, nargs='*', default=[6])
    r.add_argument('--min-tree-dist', type=float, default=6.0)
    r.add_argument('--graft-within', type=float, default=2.0)
    r.add_argument('--drop-smaller-than', type=int, default=1)
    r.add_argument('--step', type=float, default=0.25)
    r.add_argument('--smooth-coef', type=float, default=0.8)
    r.add_argument('--z-scale', type=float, default=1.0,
                   help='z_um = z_raw * z_scale + z_offset, applied on load')
    r.add_argument('--z-offset', type=float, default=0.0)
    r.add_argument('--merge-unreviewed', action='store_true',
                   help='merge regions still marked review instead of skipping')
    r.add_argument('--apply-region-fixes', action='store_true',
                   help='apply the reviewed actions in regions.csv. Off by '
                        'default: the detectors are advisory and you fix the '
                        'skeleton by hand in Blender instead.')

    args = p.parse_args()
    out_dir = Path(args.out_dir)

    if args.cmd == 'audit':
        audit(args.swc, out_dir, soma_type=args.soma_type,
              min_tree_dist=args.min_tree_dist,
              strip_types=tuple(args.strip_types), mesh=args.mesh)
        return 0

    bands = []
    for b in args.band:
        parts = b.split(':')
        if len(parts) < 2:
            sys.exit(f"bad --band '{b}', want ZMIN:ZMAX[:LABEL]")
        bands.append((float(parts[0]), float(parts[1]),
                      parts[2] if len(parts) > 2 else 'band'))
    manifest = args.manifest
    if manifest is None and (out_dir / 'regions.csv').exists():
        manifest = out_dir / 'regions.csv'
    res = repair(args.swc, args.out or (out_dir / 'repaired.swc'),
                 manifest=manifest, mesh=args.mesh, bands=bands,
                 soma_type=args.soma_type, strip_types=tuple(args.strip_types),
                 min_tree_dist=args.min_tree_dist,
                 graft_within=args.graft_within,
                 drop_smaller_than=args.drop_smaller_than,
                 step=args.step, smooth_coef=args.smooth_coef,
                 z_scale=args.z_scale, z_offset=args.z_offset,
                 unreviewed_action='merge' if args.merge_unreviewed else 'skip',
                 apply_region_fixes=args.apply_region_fixes)
    return 0 if res.get('ok') else 1


if __name__ == '__main__':
    sys.exit(main())
