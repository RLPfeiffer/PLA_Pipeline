#!/usr/bin/env python3
"""
Depth plots for AII input: bipolar cell synapses, and gap junctions by partner.

The compartment boundary is drawn, because it is the denominator density.py
uses. A partner class confined to one stratum has to be divided by that
stratum's length: lobular dendrite never receives CBb input, so pooling the
whole cell divides 48 arboreal CBb contacts by 603 um instead of 415 and
understates the density by 27%.

Both are normalised by **dendritic path length in each depth bin**, not plotted
as raw counts. Dendrite is not distributed evenly through the IPL - an AII has
two strata with a sparse gap between - so a raw count profile mostly shows where
the dendrite is, not where the input is. Dividing by length in the same bin is
the whole point of the pipeline.

Two deliberate differences from the main density run:

  1. Gap junctions are NOT filtered by Direction. A gap junction is
     bidirectional, and in this export the same partner class appears under both
     labels (GAC Aii: 25 post, 17 pre). Filtering to post would silently discard
     a third of them. The direction column is an annotation convention here, not
     biology.

  2. Synapse types are selected per plot rather than globally: chemical input
     for the BC plot, gap-junction types for the GJ plot. So this reads the raw
     synapse CSV rather than the already-filtered synapses.csv.

    python plots.py repaired.swc synapses.csv --out-dir out/192 --z-split 58.73
"""

import argparse
import collections
import csv
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")          # no display inside a pipeline run
import matplotlib.pyplot as plt

import swcrepair as sr

try:
    from density import classify_contact as classify
    from density import normalize_label as fold_label
except ImportError:                     # standalone use
    def classify(type_value, classes=None):
        t = (type_value or "").strip()
        for name, pats in (classes or [("gap_junction", ["GapJunction"]),
                                       ("psd", ["RibbonPost", "ConvPost",
                                                "Post"]),
                                       ("presynaptic", ["Pre"])]):
            if any(p in t for p in pats):
                return name
        return "other"

    def fold_label(value, aliases=None):
        v = (value or "").strip()
        return ({"CBbwf": "CBb"} if aliases is None else aliases).get(v, v)


# Partner classes counted as bipolar cell input, in plotting order.
BC_LABELS = ["RodBC", "CBa", "CBb", "BC"]
# Chemical synapse types that represent input onto this cell.
BC_TYPES = ["RibbonPost", "ConvPost"]
# Gap junction types. Substring match, so AnnularGapJunction is included.
GJ_TYPES = ["GapJunction"]

PALETTE = {
    "RodBC":    "#2a78d6",
    "CBb":      "#1d9a6c",
    "CBa":      "#e0662f",
    "BC":       "#8c6bb1",
    "GAC Aii":  "#c0392b",
    "AC":       "#7f8c8d",
    "yAC":      "#b8a038",
    "unlabeled": "#b0b0b0",
}


def colour(label, i=0):
    if label in PALETTE:
        return PALETTE[label]
    return plt.cm.tab10(i % 10)


# --------------------------------------------------------------- length by Z

def length_by_depth(skel, edges, soma_centre=None, soma_radius=0.0):
    """Dendritic path length falling in each depth bin.

    Segments are split at bin boundaries rather than assigned whole to the bin
    of their midpoint, so a long proximal segment contributes to every bin it
    crosses. Anything inside the soma sphere is excluded, matching density.py.
    """
    out = np.zeros(len(edges) - 1)
    for i, p in skel.par.items():
        if p == -1:
            continue
        a, b = np.asarray(skel.pos[p]), np.asarray(skel.pos[i])
        if soma_radius and soma_centre is not None:
            da = np.linalg.norm(a - soma_centre)
            db = np.linalg.norm(b - soma_centre)
            if da <= soma_radius and db <= soma_radius:
                continue
            if da <= soma_radius or db <= soma_radius:
                t = (soma_radius - da) / (db - da) if abs(db - da) > 1e-12 else 0.0
                t = float(np.clip(t, 0.0, 1.0))
                cross = a + t * (b - a)
                if da <= soma_radius:
                    a = cross
                else:
                    b = cross
        seg = float(np.linalg.norm(b - a))
        if seg <= 0:
            continue
        # walk the segment in small steps and drop each step in its bin
        n = max(1, int(np.ceil(abs(b[2] - a[2]) / max(edges[1] - edges[0], 1e-9) * 4)))
        for k in range(n):
            p0 = a + (b - a) * (k / n)
            p1 = a + (b - a) * ((k + 1) / n)
            zc = 0.5 * (p0[2] + p1[2])
            j = np.searchsorted(edges, zc) - 1
            if 0 <= j < len(out):
                out[j] += seg / n
    return out


# ------------------------------------------------------------ synapse loading

def load_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def pick(rows, labels=None, types=None, direction=None, label_col="NeuronLabel",
         type_col="SynapseType", dir_col="Direction", aliases=None):
    """Select rows by partner label, synapse type substring, and direction.

    Labels are folded through the alias map first, so CBbwf is selected by
    asking for CBb.
    """
    out = []
    for r in rows:
        lab = fold_label(r.get(label_col), aliases) or "unlabeled"
        typ = (r.get(type_col) or "").strip()
        if labels is not None and lab not in labels:
            continue
        if types is not None and not any(t in typ for t in types):
            continue
        if direction is not None and (r.get(dir_col) or "").strip() != direction:
            continue
        out.append(r)
    return out


def zs_of(rows, z_col="SynapseZ_um"):
    z = []
    for r in rows:
        try:
            z.append(float(r[z_col]))
        except (KeyError, TypeError, ValueError):
            pass
    return np.array(z)


# ------------------------------------------------------------------- plotting

def _depth_axes(ax, edges, z_split=0.0, label=True):
    """Depth on the vertical axis, increasing downward as in a retinal section.

    The compartment boundary is drawn because it is the denominator used by
    density.py: a partner class confined to one stratum has to be divided by
    that stratum's length, not the whole cell's.
    """
    ax.set_ylim(edges[-1], edges[0])
    if z_split:
        ax.axhline(z_split, color="#444444", lw=1.1, ls="--", zorder=1)
        if label:
            ax.text(ax.get_xlim()[1], z_split, " z_split", va="center",
                    fontsize=7, color="#444444")
    return 0.5 * (edges[:-1] + edges[1:])


def _series_by_partner(rows, contact_class, min_n, classes=None,
                       label_col="NeuronLabel", type_col="SynapseType",
                       aliases=None):
    """Partner classes carrying this contact class, most numerous first."""
    sel = [r for r in rows
           if classify(r.get(type_col), classes) == contact_class]
    by = collections.Counter(
        fold_label(r.get(label_col), aliases) or "unlabeled" for r in sel)
    order = [lab for lab, n in by.most_common() if n >= min_n]
    minor = sum(n for lab, n in by.items() if n < min_n)
    return sel, order, by, minor


def plot_depth_by_partner(skel, rows, edges, out_png, contact_class,
                          title, cell_id="", z_split=0.0, soma_centre=None,
                          soma_radius=0.0, classes=None, min_n=1,
                          only_labels=None, xmax=None):
    """Depth profile of one contact class, one series per partner class.

    One figure per contact class, with identical axes and colours, so a partner
    that makes both kinds of contact can be compared across figures directly -
    CBb has 40 gap junctions and 8 ribbons on cell 192's arboreal dendrites,
    and those are different things.
    """
    L = length_by_depth(skel, edges, soma_centre, soma_radius)
    centres = 0.5 * (edges[:-1] + edges[1:])
    sel, order, by, minor = _series_by_partner(rows, contact_class, min_n,
                                               classes)
    dropped_labels = []
    if only_labels:
        dropped_labels = [(lab, by[lab]) for lab in order
                          if lab not in only_labels]
        order = [lab for lab in order if lab in only_labels]
        sel = [r for r in sel
               if (fold_label(r.get("NeuronLabel")) or "unlabeled")
               in only_labels]

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 5.2),
                             gridspec_kw=dict(width_ratios=[0.9, 1.4, 1.4, 1.1]))

    ax = axes[0]
    ax.barh(centres, L, height=np.diff(edges) * 0.9, color="#cfd6dd",
            edgecolor="none")
    ax.set_xlabel("dendrite length\nper bin (µm)")
    ax.set_ylabel("volume Z (µm)")
    ax.set_title("dendrite available", fontsize=9)
    _depth_axes(ax, edges, z_split)

    ax = axes[1]
    for i, lab in enumerate(order):
        z = zs_of(pick(sel, labels=[lab]))
        n, _ = np.histogram(z, bins=edges)
        ax.step(n, centres, where="mid", color=colour(lab, i), lw=1.6,
                label=f"{lab} (n={by[lab]})")
    ax.set_xlabel("contacts per bin")
    ax.set_title("count by partner", fontsize=9)
    _depth_axes(ax, edges, z_split)
    if order:
        ax.legend(fontsize=7, frameon=False, loc="lower right")

    ax = axes[2]
    for i, lab in enumerate(order):
        z = zs_of(pick(sel, labels=[lab]))
        n, _ = np.histogram(z, bins=edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            dens = np.where(L > 0.5, n / L, np.nan)
        ax.step(dens, centres, where="mid", color=colour(lab, i), lw=1.8)
    ax.set_xlabel("contacts per µm of dendrite")
    ax.set_title("density (bins with <0.5 µm blanked)", fontsize=9)
    if xmax:
        ax.set_xlim(0, xmax)
    _depth_axes(ax, edges, z_split)

    ax = axes[3]
    lo_len = L[centres < z_split].sum() if z_split else 0.0
    hi_len = L[centres >= z_split].sum() if z_split else L.sum()
    lo, hi = [], []
    for lab in order:
        z = zs_of(pick(sel, labels=[lab]))
        lo.append(int((z < z_split).sum()) if z_split else 0)
        hi.append(int((z >= z_split).sum()) if z_split else int(z.size))
    y = np.arange(len(order))
    ax.barh(y, lo, color="#d8a44f", label=f"lobular ({lo_len:.0f} µm)")
    ax.barh(y, hi, left=lo, color="#4d86c6", label=f"arboreal ({hi_len:.0f} µm)")
    for k, (a, b) in enumerate(zip(lo, hi)):
        da = a / lo_len if lo_len > 0 else 0.0
        db = b / hi_len if hi_len > 0 else 0.0
        ax.text(a + b, k, f"  {da:.3f} / {db:.3f}", va="center", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    if order:
        ax.set_xlim(0, max(a + b for a, b in zip(lo, hi)) * 1.75)
    ax.set_xlabel("contacts")
    ax.set_title("total by compartment\n(lobular / arboreal per µm)",
                 fontsize=9)
    if order:
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")

    note = f"{len(sel)} contacts"
    if minor:
        note += f"   ·   {minor} in partner classes with n<{min_n} omitted"
    if dropped_labels:
        note += ("   ·   partners not shown: "
                 + ", ".join(f"{l} ({n})" for l, n in dropped_labels))
    if contact_class == "gap_junction":
        note += "   ·   both Direction labels counted, a gap junction is " \
                "bidirectional"
    fig.suptitle(f"{title} on cell {cell_id}\n{note}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return dict(length=L, centres=centres, order=order, counts=dict(by),
                hist={lab: np.histogram(zs_of(pick(sel, labels=[lab])),
                                        bins=edges)[0] for lab in order})


def plot_psd_vs_gj(skel, rows, edges, out_png, cell_id="", z_split=0.0,
                   soma_centre=None, soma_radius=0.0, classes=None, min_n=1,
                   label_filter=None):
    """Same partners, PSD against gap junction, so the two are comparable.

    A partner class can appear in both - electrical and chemical contact with
    the same cell type are separate observations, not one.
    """
    L = length_by_depth(skel, edges, soma_centre, soma_radius)
    centres = 0.5 * (edges[:-1] + edges[1:])
    lo_len = L[centres < z_split].sum() if z_split else 0.0
    hi_len = L[centres >= z_split].sum() if z_split else L.sum()

    tallies = {}
    for cc in ("psd", "gap_junction"):
        sel, _, by, _ = _series_by_partner(rows, cc, 1, classes)
        keep = (label_filter or {}).get(cc)
        if keep:
            sel = [r for r in sel
                   if (fold_label(r.get("NeuronLabel")) or "unlabeled")
                   in keep]
            by = collections.Counter({k: v for k, v in by.items() if k in keep})
        tallies[cc] = (sel, by)
    labs = sorted({lab for _, by in tallies.values() for lab in by
                   if by[lab] >= min_n},
                  key=lambda l: -(tallies["psd"][1][l]
                                  + tallies["gap_junction"][1][l]))

    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.2, 0.5 * len(labs) + 2.4)))
    w = 0.38
    y = np.arange(len(labs))

    ax = axes[0]
    ax.barh(y - w / 2, [tallies["psd"][1][l] for l in labs], height=w,
            color="#2a6fb8", label="PSD")
    ax.barh(y + w / 2, [tallies["gap_junction"][1][l] for l in labs], height=w,
            color="#c85a1e", label="gap junction")
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("contacts")
    ax.set_title("count per partner", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for off, cc, col in ((-w / 2, "psd", "#2a6fb8"),
                         (w / 2, "gap_junction", "#c85a1e")):
        sel, _ = tallies[cc]
        vals = []
        for l in labs:
            z = zs_of(pick(sel, labels=[l]))
            n_hi = int((z >= z_split).sum()) if z_split else int(z.size)
            vals.append(n_hi / hi_len if hi_len > 0 else 0.0)
        ax.barh(y + off, vals, height=w, color=col)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("contacts per µm of arboreal dendrite")
    ax.set_title(f"arboreal density ({hi_len:.0f} µm)", fontsize=9)

    fig.suptitle(f"PSD vs gap junction per partner, cell {cell_id}\n"
                 f"partner classes with fewer than {min_n} of either omitted",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return dict(labels=labs)


def read_all_summary(path):
    """Rows of all_summary.csv as dicts with numbers parsed."""
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                r["n"] = int(r["n"])
                r["length_um"] = float(r["length_um"])
                r["density_per_um"] = (float(r["density_per_um"])
                                       if r["density_per_um"] else 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            out.append(r)
    return out


def aggregate_across_cells(rows, volume):
    """Mean, SD, SEM and n cells per (compartment, contact_class, category).

    Two things this gets right that are easy to get wrong.

    It averages the per-cell DENSITY rather than dividing pooled counts by
    pooled length. Pooling weights by dendrite length and lets one large cell
    dominate; the cell is the experimental unit, so each contributes one value.

    It fills a ZERO for any cell that has no row for a category. density.py
    only emits rows for categories a cell actually has, so a cell with no CBa
    gap junctions is absent rather than zero - and averaging over only the
    cells that had some inflates the mean. With n=4 and one cell lacking a
    category that is a 33% overestimate.
    """
    vol_rows = [r for r in rows if r.get("volume") == volume]
    all_cells = sorted({r["cell_id"] for r in vol_rows})
    # compartments a cell was actually measured in, so a missing compartment is
    # not confused with a missing category
    measured = collections.defaultdict(set)
    for r in vol_rows:
        measured[r["compartment"]].add(r["cell_id"])

    groups = collections.defaultdict(dict)
    for r in vol_rows:
        key = (r["compartment"], r.get("contact_class", ""), r["category"])
        groups[key][r["cell_id"]] = r["density_per_um"]

    out = {}
    for key, per_cell in groups.items():
        comp = key[0]
        cells = sorted(measured[comp] or all_cells)
        vals = [per_cell.get(cid, 0.0) for cid in cells]
        nonzero = [cid for cid in cells if per_cell.get(cid, 0.0) > 0]
        v = np.array(vals, dtype=float)
        n = v.size
        sd = float(v.std(ddof=1)) if n > 1 else float("nan")
        out[key] = dict(mean=float(v.mean()), sd=sd,
                        sem=(sd / np.sqrt(n)) if n > 1 else float("nan"),
                        n_cells=n, n_nonzero=len(nonzero), values=v,
                        cells=cells)
    return out


def plot_project_summary(all_summary, out_dir, volume, min_cells=1,
                         log=print):
    """One figure per volume: density by category, compartment and contact class.

    Error bars are SEM across cells, with the individual cell values overlaid
    as points, because at n=5 the spread matters more than the bar. A category
    measured in only one cell gets no error bar and is marked.
    """
    rows = read_all_summary(all_summary)
    agg = aggregate_across_cells(rows, volume)
    if not agg:
        log(f"  no rows for volume {volume!r}")
        return None

    classes = [c for c in ("psd", "gap_junction", "presynaptic", "other")
               if any(k[1] == c for k in agg)]
    comps = [c for c in ("lobular", "arboreal", "cell")
             if any(k[0] == c for k in agg)]
    n_cells_total = len({r["cell_id"] for r in rows
                         if r.get("volume") == volume})

    fig, axes = plt.subplots(len(classes), 1,
                             figsize=(11, 3.1 * len(classes) + 1.0),
                             squeeze=False)
    comp_col = {"lobular": "#d8a44f", "arboreal": "#4d86c6", "cell": "#6b8f71"}

    for ax, cc in zip(axes[:, 0], classes):
        cats = sorted({k[2] for k in agg if k[1] == cc and k[2] != "ALL"})
        cats = cats + ["ALL"] if any(k[1] == cc and k[2] == "ALL"
                                     for k in agg) else cats
        x = np.arange(len(cats))
        w = 0.8 / max(len(comps), 1)
        for j, comp in enumerate(comps):
            off = (j - (len(comps) - 1) / 2) * w
            means = [agg.get((comp, cc, c), {}).get("mean", 0.0) for c in cats]
            sems = [agg.get((comp, cc, c), {}).get("sem", float("nan"))
                    for c in cats]
            ax.bar(x + off, means, width=w * 0.92, color=comp_col.get(comp),
                   label=comp, zorder=2)
            ax.errorbar(x + off, means, yerr=sems, fmt="none",
                        ecolor="#333333", elinewidth=1.0, capsize=3, zorder=3)
            for k, c in enumerate(cats):
                e = agg.get((comp, cc, c))
                if not e:
                    continue
                if e["n_cells"] > 1:
                    ax.scatter(np.full(e["n_cells"], x[k] + off),
                               e["values"], s=9, color="#222222",
                               zorder=4, linewidths=0)
                else:
                    ax.text(x[k] + off, e["mean"], "n=1", ha="center",
                            va="bottom", fontsize=6, color="#666666")
                if 0 < e["n_nonzero"] < e["n_cells"]:
                    ax.text(x[k] + off, 0, f"{e['n_nonzero']}/{e['n_cells']}",
                            ha="center", va="bottom", fontsize=5.5,
                            color="#777777", rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("contacts per µm")
        ax.set_title(cc.replace("_", " "), fontsize=10, loc="left")
        ax.grid(axis="y", color="#e8e8e8", zorder=0)
        ax.set_axisbelow(True)
        if ax is axes[0, 0]:
            ax.legend(fontsize=8, frameon=False, ncol=len(comps))

    fig.suptitle(f"Volume {volume}: contact density per µm of dendrite\n"
                 f"{n_cells_total} cell(s); bars are the mean across cells "
                 f"with zeros included, error bars SEM, points individual "
                 f"cells.\nA fraction under a bar is how many cells had any "
                 f"of that contact",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.09 / len(classes)))
    png = os.path.join(str(out_dir), f"summary_{volume}.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    log(f"  wrote {png}")

    stats = os.path.join(str(out_dir), f"summary_{volume}_stats.csv")
    with open(stats, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["volume", "compartment", "contact_class", "category",
                    "n_cells", "n_cells_nonzero", "mean_density_per_um",
                    "sd", "sem", "cells"])
        for (comp, cc, cat), e in sorted(agg.items()):
            w.writerow([volume, comp, cc, cat, e["n_cells"], e["n_nonzero"],
                        round(e["mean"], 6),
                        "" if e["n_cells"] < 2 else round(e["sd"], 6),
                        "" if e["n_cells"] < 2 else round(e["sem"], 6),
                        " ".join(e["cells"])])
    log(f"  wrote {stats}")
    return dict(png=png, stats=stats, n_cells=n_cells_total)


def plot_all_volumes(all_summary, out_dir, log=print):
    """A separate figure per volume - they are not comparable across volumes.

    Section thickness, staining and the calibration of smooth_coef are all
    volume-level properties, so pooling volumes would compare numbers derived
    on different scales.
    """
    rows = read_all_summary(all_summary)
    vols = sorted({r.get("volume", "") for r in rows if r.get("volume")})
    if not vols:
        vols = [""]
    out = []
    for v in vols:
        res = plot_project_summary(all_summary, out_dir, v, log=log)
        if res:
            out.append(res)
    return out


def make_plots(swc_path, csv_path, out_dir, z_split=0.0, bin_um=2.0,
               exclude_soma_radius=-1.0, cell_id="", contact_classes=None,
               label_filter=None, min_n=1, log=print):
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    skel = sr.Skel.load(swc_path)
    rows = load_rows(csv_path)

    soma_centre = soma_radius = None
    if skel.roots():
        root = min(skel.roots())
        soma_centre = np.asarray(skel.pos[root])
        soma_radius = (float(skel.rad[root]) if exclude_soma_radius < 0
                       else float(exclude_soma_radius))

    zs = np.array([skel.pos[i][2] for i in skel.pos])
    lo = np.floor(zs.min() / bin_um) * bin_um
    hi = np.ceil(zs.max() / bin_um) * bin_um
    edges = np.arange(lo, hi + bin_um * 0.5, bin_um)
    log(f"  depth bins {bin_um} um, Z {lo:.1f}..{hi:.1f} ({len(edges)-1} bins)")

    tally = collections.Counter(classify(r.get("SynapseType"), contact_classes)
                                for r in rows)
    log("  contact classes in the CSV: "
        + ", ".join(f"{k}={v}" for k, v in tally.most_common()))

    # one figure per contact class, shared axes so they are comparable
    titles = {"psd": "Postsynaptic densities",
              "gap_junction": "Gap junctions",
              "presynaptic": "Presynaptic contacts",
              "other": "Other annotated contacts"}
    written, results = [], {}
    for cc in ("psd", "gap_junction", "presynaptic"):
        if not tally.get(cc, 0):
            continue
        keep = (label_filter or {}).get(cc)
        png = os.path.join(out_dir, f"depth_{cc}.png")
        title = titles.get(cc, cc)
        if keep:
            title += f" from {'/'.join(keep)}"
        results[cc] = plot_depth_by_partner(
            skel, rows, edges, png, cc, title, cell_id, z_split,
            soma_centre, soma_radius, classes=contact_classes, min_n=min_n,
            only_labels=keep)
        log(f"  wrote {png}")
        written.append(png)

    cmp_png = os.path.join(out_dir, "psd_vs_gapjunction.png")
    plot_psd_vs_gj(skel, rows, edges, cmp_png, cell_id, z_split,
                   soma_centre, soma_radius, classes=contact_classes,
                   min_n=min_n, label_filter=label_filter)
    log(f"  wrote {cmp_png}")
    written.append(cmp_png)

    # the numbers behind every figure
    data = os.path.join(out_dir, "depth_profile.csv")
    Lb = length_by_depth(skel, edges, soma_centre, soma_radius)
    centres = 0.5 * (edges[:-1] + edges[1:])
    cols = []
    for cc, res in results.items():
        for lab in res["order"]:
            cols.append((cc, lab, res["hist"][lab]))
    with open(data, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell_id", "z_centre_um", "dendrite_length_um"]
                   + [f"{cc}__{lab}" for cc, lab, _ in cols])
        for k in range(len(centres)):
            w.writerow([cell_id, round(float(centres[k]), 3),
                        round(float(Lb[k]), 4)]
                       + [int(h[k]) for _, _, h in cols])
    log(f"  wrote {data}")
    return dict(figures=written, data_csv=data,
                counts={cc: dict(r["counts"]) for cc, r in results.items()})


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("swc")
    p.add_argument("csv")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--z-split", type=float, default=0.0,
                   help="compartment boundary in volume Z. Drawn on the "
                        "profiles and used to split the per-partner totals")
    p.add_argument("--cell-id", default="")
    p.add_argument("--bin-um", type=float, default=2.0)
    p.add_argument("--exclude-soma-radius", type=float, default=-1.0)
    p.add_argument("--min-n", type=int, default=1,
                   help="omit partner classes with fewer than this many "
                        "contacts. Default 1 shows all of them.")
    p.add_argument("--psd-labels", default=None,
                   help="comma-separated partner classes to keep in the PSD "
                        "figure, e.g. RodBC,CBa,CBb,BC. Gap junctions are "
                        "never restricted.")
    a = p.parse_args()
    try:
        make_plots(a.swc, a.csv, a.out_dir, z_split=a.z_split, bin_um=a.bin_um,
                   exclude_soma_radius=a.exclude_soma_radius,
                   cell_id=a.cell_id, min_n=a.min_n,
                   label_filter=({"psd": [x.strip() for x in
                                          a.psd_labels.split(",")]}
                                 if a.psd_labels else None))
    except Exception as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
