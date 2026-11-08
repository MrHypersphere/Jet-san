"""
plot_jets.py

Quark vs gluon comparison plots from a jet file written by make_jets.py.

Writes two figures:
  <prefix>_observables.png  distributions of each jet-shape observable,
                            quark vs gluon, pT-reweighted (normalized to unit area)
  <prefix>_separation.png   how well each observable alone separates quark from
                            gluon jets (ROC AUC: 0.5 = no separation, 1 = perfect)

Usage:
    python plot_jets.py jets_clean.parquet --prefix plots/clean
"""
import argparse
import os

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

QUARK = "#2a78d6"
GLUON = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# (column, axis label, which class is expected higher)
OBSERVABLES = [
    ("n_const", "Constituent count  ($n_{\\mathrm{const}}$)", "gluon"),
    ("girth", "Girth  ($g$)", "gluon"),
    ("ptd", "Momentum dispersion  ($p_T^D$)", "quark"),
    ("mass", "Jet mass  ($m$)  [GeV]", "gluon"),
    ("e2", "Energy correlation  ($e_2^{(\\beta=1)}$)", "gluon"),
]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def weighted_auc(x, label, w):
    """Area under the ROC curve for 'larger x means gluon', with weights.
    Equals the probability that a random gluon jet has larger x than a random
    quark jet (ties count half)."""
    order = np.argsort(x, kind="mergesort")
    x, label, w = x[order], label[order], w[order]
    wg = np.where(label == 1, w, 0.0)
    wq = np.where(label == 0, w, 0.0)
    # group ties
    uniq, start = np.unique(x, return_index=True)
    gq = np.add.reduceat(wq, start)
    gg = np.add.reduceat(wg, start)
    q_below = np.cumsum(gq) - gq
    auc = np.sum(gg * (q_below + 0.5 * gq))
    return float(auc / (gg.sum() * gq.sum()))


def bins_for(col, x):
    lo, hi = np.percentile(x, [0.0, 99.5])
    if col == "n_const":
        step = max(1, int((hi - lo) // 45) + 1)
        return np.arange(np.floor(lo) - 0.5, hi + 1.5, step)
    return np.linspace(lo, hi, 45)


def plot_observables(jets, path, title_extra):
    lab = ak.to_numpy(jets.label)
    w = ak.to_numpy(jets.weight).astype(float)
    pt = ak.to_numpy(jets.pt_sub).astype(float)
    q, g = lab == 0, lab == 1

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor=SURFACE)
    axes = axes.ravel()

    for ax, (col, name, higher) in zip(axes, OBSERVABLES):
        x = ak.to_numpy(jets[col]).astype(float)
        bins = bins_for(col, x)
        for m, color in ((q, QUARK), (g, GLUON)):
            ax.hist(x[m], bins=bins, weights=w[m], density=True, histtype="step",
                    lw=2, color=color)
        mq = np.average(x[q], weights=w[q])
        mg = np.average(x[g], weights=w[g])
        style_axes(ax)
        ax.set_xlabel(name, color=INK, fontsize=10.5)
        ax.set_ylabel("fraction of jets (normalized)", color=INK2, fontsize=9)
        ax.text(0.98, 0.95, f"mean  quark {mq:.3g}\n          gluon {mg:.3g}\n{higher} higher",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2,
                family="monospace")

    # last panel: jet pT before and after reweighting
    ax = axes[5]
    bins = np.linspace(100, 400, 46)
    ax.hist(pt[q], bins=bins, density=True, histtype="step", lw=2, color=QUARK)
    ax.hist(pt[g], bins=bins, density=True, histtype="step", lw=1.5, color=GLUON, ls=(0, (3, 2)))
    ax.hist(pt[g], bins=bins, weights=w[g], density=True, histtype="step", lw=2, color=GLUON)
    ax.set_yscale("log")
    style_axes(ax)
    ax.set_xlabel("Jet transverse momentum  ($p_T$)  [GeV]", color=INK, fontsize=10.5)
    ax.set_ylabel("fraction of jets (log)", color=INK2, fontsize=9)
    ax.text(0.98, 0.95, "dashed: gluon before reweighting\nsolid: gluon after (matches quark)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)

    handles = [plt.Line2D([], [], color=QUARK, lw=2.5), plt.Line2D([], [], color=GLUON, lw=2.5)]
    fig.legend(handles, [f"quark jets ({q.sum():,})", f"gluon jets ({g.sum():,})"],
               loc="upper right", ncol=2, frameon=False, fontsize=11, labelcolor=INK)
    fig.suptitle("Quark vs gluon jet shapes" + title_extra, x=0.02, ha="left",
                 fontsize=15, color=INK, fontweight="bold")
    fig.text(0.02, 0.925, "anti-kT R = 0.4, |η| < 2.5, jets reweighted so quark and gluon share one pT spectrum",
             fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {path}")


def plot_separation(jets, path, title_extra):
    lab = ak.to_numpy(jets.label)
    w = ak.to_numpy(jets.weight).astype(float)
    keep = w > 0
    rows = []
    for col, name, higher in OBSERVABLES:
        x = ak.to_numpy(jets[col]).astype(float)
        sign = 1.0 if higher == "gluon" else -1.0
        rows.append((name, weighted_auc(sign * x[keep], lab[keep], w[keep])))
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, [v - 0.5 for v in vals], left=0.5, height=0.6, color=INK2)
    for yi, v in zip(y, vals):
        ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(y, names, color=INK, fontsize=10)
    ax.set_xlim(0.5, 1.0)
    style_axes(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_xlabel("ROC AUC  (0.5 = no separation, 1.0 = perfect)", color=INK2, fontsize=10)
    fig.suptitle("How well each observable alone separates quark from gluon jets",
                 x=0.02, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.text(0.02, 0.885, "pT-reweighted" + title_extra.replace(" (", ", ").rstrip(")"),
             fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {path}")
    for n, v in reversed(rows):
        print(f"  AUC {v:.3f}  {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jets", help="jet parquet file from make_jets.py")
    ap.add_argument("--prefix", default=None, help="output path prefix (default: input name)")
    ap.add_argument("--title", default="", help="extra text appended to figure titles")
    args = ap.parse_args()

    prefix = args.prefix or os.path.splitext(args.jets)[0]
    if os.path.dirname(prefix):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
    cols = ["label", "weight", "pt_sub"] + [c for c, _, _ in OBSERVABLES]
    jets = ak.from_parquet(args.jets, columns=cols)
    extra = f" ({args.title})" if args.title else ""
    plot_observables(jets, prefix + "_observables.png", extra)
    plot_separation(jets, prefix + "_separation.png", extra)


if __name__ == "__main__":
    main()
