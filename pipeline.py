#!/usr/bin/env python3
"""
Multi-cell pipeline driver.

    pipeline.py init   PROJECT                     scaffold config + input dirs
    pipeline.py paths  PROJECT                     show resolved paths, flag typos
    pipeline.py discover PROJECT                   list matched cells, write cells.csv
    pipeline.py audit  PROJECT [--cells 192 476]   report + regions.csv per cell
    pipeline.py repair PROJECT [--cells ...]       apply fixes, write repaired.swc
    pipeline.py density PROJECT [--cells ...]      per-branch + summary CSVs
    pipeline.py run    PROJECT                     audit, repair, density, aggregate
    pipeline.py aggregate PROJECT                  concatenate per-cell CSVs

Paths come from PROJECT/cells.toml. Per-cell parameters come from
PROJECT/cells.csv, where a blank cell means "use the default from the toml".
Review actions live in out/<id>/regions.csv and are preserved across re-runs, so
re-auditing never discards decisions you have already made.

Typical first pass:

    pipeline.py init   ~/aii
    # edit ~/aii/cells.toml to point at your export directories
    pipeline.py paths  ~/aii
    pipeline.py discover ~/aii
    # set z_split (and smooth_coef once calibrated) in ~/aii/cells.csv
    pipeline.py audit  ~/aii
    # review out/<id>/regions.csv for each cell, in Blender via blender_qc.py
    pipeline.py run    ~/aii
"""

import argparse
import csv
import datetime
import sys
import traceback
from pathlib import Path

import cellpaths
import density as density_mod
import repair_run

try:
    import plots as plots_mod
except ImportError as _e:          # matplotlib missing
    plots_mod = None
    _plots_err = _e


# ----------------------------------------------------------------- logging

class Tee:
    """Write to stdout and to a per-cell log file at the same time."""

    def __init__(self, path, prefix='', echo=True):
        self.path = Path(path) if path else None
        self.prefix = prefix
        self.echo = echo
        self.fh = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(self.path, 'a')
            self.fh.write(f"\n===== {datetime.datetime.now().isoformat(timespec='seconds')} =====\n")

    def __call__(self, msg=''):
        if self.echo:
            print(f"{self.prefix}{msg}")
        if self.fh:
            self.fh.write(f"{msg}\n")

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None


def open_project(args):
    try:
        return cellpaths.Project(args.project)
    except FileNotFoundError as e:
        sys.exit(str(e))
    except ValueError as e:
        sys.exit(f"config error: {e}")
    except Exception as e:          # tomllib raises its own decode error type
        sys.exit(f"could not read {Path(args.project) / cellpaths.CONFIG_NAME}: "
                 f"{type(e).__name__}: {e}\n"
                 f"Windows paths need SINGLE quotes in TOML: "
                 f"swc_dir = 'E:\\Data\\Project'")


def discover_or_exit(proj, only=None):
    try:
        return proj.discover(only=only)
    except ValueError as e:
        sys.exit(f"cells.csv error: {e}")


def select(proj, args):
    cells = discover_or_exit(proj, only=args.cells)
    if not cells:
        sys.exit("no cells matched. Run `pipeline.py paths` and "
                 "`pipeline.py discover` to see why.")
    return cells


def record_run(proj, rows):
    """Append to run_manifest.csv so a batch leaves an audit trail."""
    path = proj.run_manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['timestamp', 'cell_id', 'stage', 'status', 'detail']
    new = not path.exists()
    with open(path, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ commands

def cmd_init(args):
    cfg = Path(args.project).expanduser().resolve() / cellpaths.CONFIG_NAME
    existed = cfg.exists()
    try:
        proj = cellpaths.Project.init(args.project)
    except ValueError as e:
        sys.exit(f"config error: {e}")
    print(proj.describe())
    if existed:
        print(f"\n{proj.config_path} already exists - left untouched.")
        print("That file, not any copy elsewhere, is what every command reads.")
        print("Delete it and re-run init if you want the current defaults.")
    else:
        print(f"\nwrote {proj.config_path}")
        print("Edit that file - the one just named - to point at your exports.")
    print(f"Then:  pipeline.py paths {args.project}")


def cmd_paths(args):
    proj = open_project(args)
    print(proj.describe())
    for kind, d in (('swc', proj.swc_dir), ('mesh', proj.mesh_dir),
                    ('synapse', proj.synapse_dir)):
        pats = proj.patterns.get(kind, [])
        print(f"\n{kind} patterns: {pats}")
        if not d.is_dir():
            print(f"  directory does not exist: {d}")
            continue
        matched = proj._scan(d, kind)
        print(f"  {len(matched)} file(s) matched")
    unmatched = proj.unmatched()
    if any(unmatched.values()):
        print("\nfiles present but not matched by any pattern:")
        for kind, names in unmatched.items():
            for n in names[:10]:
                print(f"  {kind:<8} {n}")
            if len(names) > 10:
                print(f"  {kind:<8} ... and {len(names)-10} more")
        print("Add the naming convention to [patterns] in cells.toml.")


def cmd_discover(args):
    proj = open_project(args)
    cells = discover_or_exit(proj, only=args.cells)
    print(proj.describe())
    print(f"\n{len(cells)} cell(s):")
    for c in cells:
        miss = c.missing()
        flag = f"   MISSING {','.join(miss)}" if miss else ""
        zs = c.params.get('z_split') or 0
        if not zs:
            zflag = "   z_split UNSET"
        elif c.params.get('_z_split_from_section'):
            zflag = (f"   z_split section "
                     f"{c.params['_z_split_from_section']:.0f} "
                     f"= {zs:.3f} um")
        else:
            zflag = f"   z_split {zs:.3f} um"
        print(f"  {c}{flag}{zflag}")
    if not cells:
        print("  none. Check `pipeline.py paths` for unmatched filenames.")
        return
    path = proj.write_param_template(cells)
    print(f"\nwrote {path}")
    print("Set z_split per cell there (blank = use the toml default). "
          "Existing values are preserved.")


def cmd_audit(args):
    proj = open_project(args)
    cells = select(proj, args)
    rows = []
    for c in cells:
        if not c.swc:
            print(f"[{c.cell_id}] no SWC, skipping")
            rows.append(dict(timestamp=_now(), cell_id=c.cell_id, stage='audit',
                             status='skipped', detail='no swc'))
            continue
        proj.ensure_out(c)
        log = Tee(c.log, prefix=f"[{c.cell_id}] ")
        try:
            res = repair_run.audit(
                c.swc, c.out_dir,
                soma_type=int(c.params.get('soma_type', 1)),
                min_tree_dist=float(c.params.get('min_tree_dist', 6.0)),
                strip_types=tuple(proj.defaults.get('strip_types', [6])),
                mesh=c.mesh, log=log)
            if not proj.repair_options()['apply_region_fixes']:
                log("")
                log("regions.csv is a REFERENCE for the Blender add-on, not a "
                    "to-do list -")
                log("repair ignores it. No need to fill in the action column.")
            rows.append(dict(timestamp=_now(), cell_id=c.cell_id, stage='audit',
                             status='ok',
                             detail=f"{len(res['regions'])} regions flagged"))
        except Exception as e:
            log(f"FAILED: {e}")
            log(traceback.format_exc())
            rows.append(dict(timestamp=_now(), cell_id=c.cell_id, stage='audit',
                             status='failed', detail=str(e)))
        finally:
            log.close()
        print()
    record_run(proj, rows)
    _tally(rows)


def cmd_repair(args):
    proj = open_project(args)
    cells = select(proj, args)
    rows = []
    for c in cells:
        if not c.swc:
            continue
        proj.ensure_out(c)
        log = Tee(c.log, prefix=f"[{c.cell_id}] ")
        try:
            bands = _bands_for(proj, c)
            ropt = proj.repair_options()
            src, kind = c.repair_input(prefer_edited=ropt['prefer_edited'])
            if kind == 'edited':
                log(f"reading your hand-edited skeleton: {src}")
            res = repair_run.repair(
                src, c.repaired_swc,
                manifest=c.regions_csv if c.regions_csv.exists() else None,
                mesh=c.mesh if (c.mesh and bands) else None,
                bands=bands,
                soma_type=int(c.params.get('soma_type', 1)),
                strip_types=tuple(proj.defaults.get('strip_types', [6])),
                min_tree_dist=float(c.params.get('min_tree_dist', 6.0)),
                graft_within=float(c.params.get('graft_within', 2.0)),
                drop_smaller_than=int(c.params.get('drop_smaller_than', 1)),
                step=float(c.params.get('step', 0.25)),
                smooth_coef=float(c.params.get('smooth_coef', 0.8)),
                z_scale=float(c.params.get('z_scale', 1.0)),
                z_offset=float(c.params.get('z_offset', 0.0)),
                unreviewed_action='merge' if args.merge_unreviewed else 'skip',
                apply_region_fixes=(args.apply_region_fixes
                                    or ropt['apply_region_fixes']),
                log=log)
            if res.get('ok'):
                rows.append(dict(timestamp=_now(), cell_id=c.cell_id,
                                 stage='repair', status='ok',
                                 detail=f"{kind} source, "
                                        f"{res['before']['length']:.1f} -> "
                                        f"{res['after']['length']:.1f} um"))
            else:
                rows.append(dict(timestamp=_now(), cell_id=c.cell_id,
                                 stage='repair', status='failed',
                                 detail=f"{len(res.get('errors',[]))} errors"))
        except Exception as e:
            log(f"FAILED: {e}")
            log(traceback.format_exc())
            rows.append(dict(timestamp=_now(), cell_id=c.cell_id, stage='repair',
                             status='failed', detail=str(e)))
        finally:
            log.close()
        print()
    record_run(proj, rows)
    _tally(rows)


def cmd_density(args):
    proj = open_project(args)
    cells = select(proj, args)
    rows = []
    for c in cells:
        if not c.ready_for_density():
            why = 'no repaired.swc' if not c.repaired_swc.exists() else 'no synapse csv'
            print(f"[{c.cell_id}] skipping: {why}")
            rows.append(dict(timestamp=_now(), cell_id=c.cell_id, stage='density',
                             status='skipped', detail=why))
            continue
        log = Tee(c.log, prefix=f"[{c.cell_id}] ")
        try:
            inc, exc, bid = proj.filters()
            res = density_mod.analyse(
                c.repaired_swc, c.synapses, c.out_dir,
                z_split=float(c.params.get('z_split') or 0),
                use_compartments=(
                    str(c.params.get('use_compartments', True)).lower()
                    not in ('0', 'false', 'no')),
                max_dist=float(c.params.get('max_dist', 3.0)),
                group_by=str(c.params.get('group_by', 'type')),
                exclude_soma_radius=float(c.params.get('exclude_soma_radius', 0.0)),
                include=inc, exclude=exc, bidirectional=bid,
                contact_classes=proj.contact_classes(),
                partner_aliases=proj.partner_aliases(),
                cell_id=c.cell_id, volume=c.volume, log=log)
            density_mod.print_summary(res)
            detail = (f"{res['n_assigned']}/{res['n_synapses']} assigned, "
                      f"{100*res['frac_unassigned']:.1f}% orphan")
            rows.append(dict(timestamp=_now(), cell_id=c.cell_id, stage='density',
                             status='warn' if res['warnings'] else 'ok',
                             detail=detail))
        except Exception as e:
            log(f"FAILED: {e}")
            rows.append(dict(timestamp=_now(), cell_id=c.cell_id, stage='density',
                             status='failed', detail=str(e)))
        finally:
            log.close()
        print()
    record_run(proj, rows)
    _tally(rows)


def cmd_frames(args):
    """Compare the coordinate frame of each cell's SWC, mesh and synapse CSV.

    Run this on every new export. A mismatch here - typically an SWC reporting
    Z as a section index rather than micrometres - makes every synapse fail to
    assign, and the resulting table looks plausible rather than empty.
    """
    import numpy as np
    import swcrepair as sr
    import density as dm

    proj = open_project(args)
    problems = []
    for c in select(proj, args):
        print(f"===== cell {c.cell_id}"
              + (f" [{c.volume}]" if c.volume else "") + " =====")
        frames = {}

        if c.swc:
            sk = sr.Skel.load(c.swc)
            P = np.array([sk.pos[i] for i in sk.pos])
            z = np.unique(np.round(P[:, 2], 4))
            step = float(np.median(np.diff(z))) if z.size > 1 else float('nan')
            frames['swc'] = (P, step,
                             bool(np.allclose(P[:, 2], np.round(P[:, 2]))))
        if c.mesh:
            try:
                V, _ = sr.load_dae_mesh(c.mesh)
                frames['mesh'] = (V, float('nan'), False)
            except Exception as e:
                print(f"  mesh unreadable: {e}")
        if c.synapses:
            try:
                syns, cols, _ = dm.load_synapses(c.synapses, log=lambda *a: None)
                S = np.array([s['pos'] for s in syns])
                z = np.unique(np.round(S[:, 2], 4))
                step = float(np.median(np.diff(z))) if z.size > 1 else float('nan')
                frames['csv'] = (S, step,
                                 bool(np.allclose(S[:, 2], np.round(S[:, 2]))))
            except Exception as e:
                print(f"  synapse CSV unreadable: {e}")

        th = proj.section_thickness
        print(f"  {'source':<7} {'X range':>18} {'Y range':>18} "
              f"{'Z range':>18} {'Z step':>8} {'sections':>16}  int?")
        for name, (A, step, isint) in frames.items():
            rng = lambda i: f"{A[:, i].min():8.2f}..{A[:, i].max():8.2f}"
            st = '-' if step != step else f"{step:.4f}"
            if th:
                lo, hi = A[:, 2].min() / th, A[:, 2].max() / th
                secs = f"{lo:7.0f}..{hi:7.0f}"
            else:
                secs = '-'
            print(f"  {name:<7} {rng(0):>18} {rng(1):>18} {rng(2):>18} "
                  f"{st:>8} {secs:>16}  {'yes' if isint else 'no'}")
        if th and 'swc' in frames:
            z = frames['swc'][0][:, 2]
            dev = np.abs(z / th - np.round(z / th)).max()
            print(f"  section thickness {th} um: max deviation from an integer "
                  f"section {dev:.6f}"
                  + ("  (exact)" if dev < 1e-6 else "  (CHECK - not a clean fit)"))

        # flag a frame mismatch and suggest the fix
        if 'swc' in frames:
            sw = frames['swc'][0]
            for other in ('mesh', 'csv'):
                if other not in frames:
                    continue
                ot = frames[other][0]
                for i, ax in enumerate('XYZ'):
                    a = sw[:, i].max() - sw[:, i].min()
                    b = ot[:, i].max() - ot[:, i].min()
                    if b > 0 and not (0.5 < a / b < 2.0):
                        ratio = b / a
                        msg = (f"cell {c.cell_id}: SWC {ax} extent is "
                               f"{a:.2f} but {other} is {b:.2f} "
                               f"(factor {ratio:.4f})")
                        problems.append(msg)
                        print(f"  MISMATCH: {msg}")
                        if ax == 'Z':
                            print(f"    -> try z_scale = {ratio:.4f} in "
                                  f"cells.csv, then check the offset")
            if frames['swc'][2] and frames['swc'][1] == 1.0:
                msg = (f"cell {c.cell_id}: SWC Z is integer with step 1 - "
                       f"almost certainly a section index, not micrometres")
                problems.append(msg)
                print(f"  MISMATCH: {msg}")
                print(f"    -> set z_scale to your section thickness in um "
                      f"(0.07 for 70 nm sections)")
        print()

    if problems:
        print(f"{len(problems)} frame problem(s) found - fix z_scale / z_offset "
              f"in cells.csv before running repair")
    else:
        print("all frames consistent")


def cmd_census(args):
    """Column census of each cell's synapse CSV - run this on a new export."""
    proj = open_project(args)
    for c in select(proj, args):
        if not c.synapses:
            continue
        print(f"===== cell {c.cell_id} =====")
        density_mod.print_census(c.synapses)
        if not args.all:
            print("(pass --all to census every cell)")
            break


def cmd_plots(args):
    """Depth profiles for bipolar cell input and gap junctions, per cell."""
    proj = open_project(args)
    if plots_mod is None:
        sys.exit(f"plotting needs matplotlib: {_plots_err}\n"
                 f"  pip install matplotlib")
    rows = []
    for c in select(proj, args):
        if not (c.repaired_swc.exists() and c.synapses):
            why = ('no repaired.swc' if not c.repaired_swc.exists()
                   else 'no synapse csv')
            print(f"[{c.cell_id}] skipping: {why}")
            rows.append(dict(timestamp=_now(), cell_id=c.cell_id, stage='plots',
                             status='skipped', detail=why))
            continue
        log = Tee(c.log, prefix=f"[{c.cell_id}] ")
        try:
            _, _, bid = proj.filters()
            res = plots_mod.make_plots(
                str(c.repaired_swc), str(c.synapses), c.out_dir,
                z_split=float(c.params.get('z_split') or 0),
                bin_um=float(proj.cfg.get('plots', {}).get('bin_um', 2.0)),
                exclude_soma_radius=float(
                    c.params.get('exclude_soma_radius', -1.0)),
                cell_id=c.cell_id,
                contact_classes=proj.contact_classes(),
                min_n=int(proj.cfg.get('plots', {}).get('min_n', 1)),
                label_filter=(proj.cfg.get('plots', {}).get('labels')
                              or None),
                log=log)
            rows.append(dict(timestamp=_now(), cell_id=c.cell_id, stage='plots',
                             status='ok',
                             detail=", ".join(
                                 f"{cc}={sum(v.values())}"
                                 for cc, v in res['counts'].items())))
        except Exception as e:
            log(f"FAILED: {e}")
            log(traceback.format_exc())
            rows.append(dict(timestamp=_now(), cell_id=c.cell_id, stage='plots',
                             status='failed', detail=str(e)))
        finally:
            log.close()
        print()
    record_run(proj, rows)
    _tally(rows)


def cmd_summary(args):
    """Project-level bar charts: one per volume, cells combined with SEM."""
    proj = open_project(args)
    if plots_mod is None:
        sys.exit(f"plotting needs matplotlib: {_plots_err}")
    if not proj.all_summary.exists():
        sys.exit(f"{proj.all_summary} not found - run "
                 f"`pipeline.py aggregate {args.project}` first")
    plots_mod.plot_all_volumes(str(proj.all_summary), proj.out_dir,
                               log=lambda m: print(m))


def cmd_aggregate(args):
    proj = open_project(args)
    cells = discover_or_exit(proj, only=args.cells)
    for name, attr in (('all_branches.csv', 'branches_csv'),
                       ('all_summary.csv', 'summary_csv')):
        target = proj.out_dir / name
        header, out_rows = None, []
        for c in cells:
            src = getattr(c, attr)
            if not src.exists():
                continue
            with open(src, newline='') as fh:
                rdr = csv.reader(fh)
                h = next(rdr, None)
                if h is None:
                    continue
                if header is None:
                    header = h
                elif h != header:
                    # different category columns between cells; union them
                    header = _union_header(header, h)
                for row in rdr:
                    out_rows.append(dict(zip(h, row)))
        if header is None:
            print(f"nothing to aggregate into {name}")
            continue
        with open(target, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=header, restval='')
            w.writeheader()
            for r in out_rows:
                w.writerow({k: r.get(k, '') for k in header})
        print(f"wrote {target}  ({len(out_rows)} rows from "
              f"{sum(1 for c in cells if getattr(c, attr).exists())} cell(s))")

    if plots_mod is not None and proj.all_summary.exists():
        try:
            plots_mod.plot_all_volumes(str(proj.all_summary), proj.out_dir,
                                       log=lambda m: print(m))
        except Exception as e:
            print(f"project summary plot skipped: {e}")
            print(traceback.format_exc())


def cmd_run(args):
    cmd_audit(args)
    cmd_repair(args)
    cmd_density(args)
    if plots_mod is not None:
        cmd_plots(args)
    else:
        print("skipping plots: matplotlib not installed")
    cmd_aggregate(args)


# ------------------------------------------------------------------ helpers

def _now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def _union_header(a, b):
    out = list(a)
    for col in b:
        if col not in out:
            out.append(col)
    return out


def _bands_for(proj, cell):
    """Validation bands: per-cell columns win, else the toml [validation] list."""
    bands = []
    raw = proj.cfg.get('validation', {}).get('bands', [])
    for entry in raw:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            bands.append((float(entry[0]), float(entry[1]),
                          str(entry[2]) if len(entry) > 2 else 'band'))
    return bands


def _tally(rows):
    if not rows:
        return
    counts = {}
    for r in rows:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    print("  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    for r in rows:
        if r['status'] in ('failed', 'warn'):
            print(f"  {r['status']:<8} {r['cell_id']:<8} {r['stage']:<8} {r['detail']}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name, fn, needs_cells in (
            ('init', cmd_init, False),
            ('paths', cmd_paths, False),
            ('discover', cmd_discover, True),
            ('audit', cmd_audit, True),
            ('repair', cmd_repair, True),
            ('density', cmd_density, True),
            ('frames', cmd_frames, True),
            ('census', cmd_census, True),
            ('plots', cmd_plots, True),
            ('summary', cmd_summary, True),
            ('aggregate', cmd_aggregate, True),
            ('run', cmd_run, True)):
        s = sub.add_parser(name)
        s.add_argument('project')
        if needs_cells:
            s.add_argument('--cells', nargs='*', default=None,
                           help='restrict to these cell ids')
        if name == 'census':
            s.add_argument('--all', action='store_true',
                           help='census every cell, not just the first')
        if name in ('repair', 'run'):
            s.add_argument('--merge-unreviewed', action='store_true',
                           help='merge duplicate regions still marked review')
            s.add_argument('--apply-region-fixes', action='store_true',
                           help='apply regions.csv actions. Off by default: '
                                'the detectors are advisory and you fix the '
                                'skeleton by hand in Blender.')
        s.set_defaults(func=fn)
    args = p.parse_args()
    if not hasattr(args, 'cells'):
        args.cells = None
    if not hasattr(args, 'merge_unreviewed'):
        args.merge_unreviewed = False
    if not hasattr(args, 'apply_region_fixes'):
        args.apply_region_fixes = False
    if not hasattr(args, 'all'):
        args.all = False
    args.func(args)


if __name__ == '__main__':
    main()
