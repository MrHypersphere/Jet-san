"""
make_jets.py

Turn the event files from generate_samples.py into a labeled quark/gluon jet
dataset. Optionally embed every event in the toy thermal background first.

Steps, per event:
  1. (optional) embed: add a thermal background regenerated from a seed
  2. cluster anti-kT R = 0.4 on constituents above --const-pt-min
  3. keep jets with pT > --jet-pt-min and |eta| < --jet-eta-max
  4. take the leading jet (Z+jet) or the two leading jets (--max-jets 2, dijets)
  5. match each kept jet to the nearest hard parton within --match-dr
  6. label: 1 = gluon (parton_id 21), 0 = quark; unmatched jets are dropped
  7. compute observables and store constituents
After all inputs: per-jet weights so quark and gluon jets share one pT spectrum.

Usage:
    # clean (vacuum) jets
    python make_jets.py --inputs quark.parquet gluon.parquet --out jets_clean.parquet

    # embedded in thermal background
    python make_jets.py --inputs quark.parquet gluon.parquet --out jets_embed.parquet --embed

    # quick look
    python make_jets.py --inputs quark.parquet gluon.parquet --out test.parquet --max-events 5000 --plot checks.png

Requires generate_samples.py in the same folder (for the background model),
plus: pip install fastjet vector awkward pyarrow numpy matplotlib
"""
import argparse
import time

import awkward as ak
import fastjet
import numpy as np
import vector

import generate_samples as gs

vector.register_awkward()

BKG_STREAM_STEP = 7919  # offsets the background seed per input file (see notes)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def dphi(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def jet_observables(pt, eta, phi, jet_pt, jet_eta, jet_phi, jet_m, beta=1.0):
    """Observables for one jet from its constituent arrays (numpy).
    girth = sum pT_i dR_i / pT_jet
    pT^D  = sqrt(sum pT_i^2) / sum pT_i
    e2    = sum_{i<j} pT_i pT_j dR_ij^beta / pT_jet^2   (normalized ECF, beta = 1)
    """
    dr = np.hypot(eta - jet_eta, dphi(phi, jet_phi))
    s = pt.sum()
    girth = float((pt * dr).sum() / jet_pt)
    ptd = float(np.sqrt((pt ** 2).sum()) / s) if s > 0 else 0.0
    if len(pt) > 1:
        deta = eta[:, None] - eta[None, :]
        dph = dphi(phi[:, None], phi[None, :])
        drij = np.hypot(deta, dph) ** beta
        e2 = float(0.5 * (pt[:, None] * pt[None, :] * drij).sum() / jet_pt ** 2)
    else:
        e2 = 0.0
    return {"mass": float(jet_m), "n_const": int(len(pt)), "girth": girth, "ptd": ptd, "e2": e2}


def rho_grid_median(pt, eta, phi, eta_max=3.0, n_eta=12, n_phi=12):
    """Event background density estimate: median pT per unit area over an
    (eta, phi) grid. The median is robust to the one or two patches a hard
    jet occupies. Patches are 0.5 x 0.52 here."""
    h, _, _ = np.histogram2d(eta, phi, bins=[n_eta, n_phi],
                             range=[[-eta_max, eta_max], [0, 2 * np.pi]], weights=pt)
    area = (2 * eta_max / n_eta) * (2 * np.pi / n_phi)
    return float(np.median(h) / area)


def build_particles(events, embed, bkg_seed_base, file_idx, bkg_kwargs):
    """Return a jagged awkward array of particles (pt, eta, phi, mass, origin)
    for this chunk. origin: 1 = Pythia signal, 0 = thermal background."""
    if not embed:
        return ak.zip(
            {"pt": events.pt, "eta": events.eta, "phi": events.phi, "mass": events.m,
             "origin": ak.ones_like(events.pt, dtype=np.int8)}
        )
    seed = bkg_seed_base + BKG_STREAM_STEP * file_idx
    out = {k: [] for k in ("pt", "eta", "phi", "mass", "origin")}
    for ev in events:
        bkg, _ = gs.thermal_background(seed, int(ev.event_id), **bkg_kwargs)
        out["pt"].append(np.concatenate([ev.pt.to_numpy(), bkg[:, 0]]))
        out["eta"].append(np.concatenate([ev.eta.to_numpy(), bkg[:, 1]]))
        out["phi"].append(np.concatenate([ev.phi.to_numpy(), bkg[:, 2]]))
        out["mass"].append(np.concatenate([ev.m.to_numpy(), bkg[:, 3]]))
        out["origin"].append(np.concatenate([np.ones(len(ev.pt), np.int8), np.zeros(len(bkg), np.int8)]))
    return ak.zip({k: ak.Array(v) for k, v in out.items()})


# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------
def process_chunk(events, args, file_idx, bkg_kwargs, stats):
    parts = build_particles(events, args.embed, args.bkg_seed, file_idx, bkg_kwargs)
    parts = parts[parts.pt > args.const_pt_min]
    vec = ak.zip({"pt": parts.pt, "eta": parts.eta, "phi": parts.phi, "mass": parts.mass},
                 with_name="Momentum4D")

    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, args.R)
    cs = fastjet.ClusterSequence(vec, jet_def)
    jets = cs.inclusive_jets(min_pt=args.jet_pt_min)
    idx = cs.constituent_index(min_pt=args.jet_pt_min)

    area = np.pi * args.R ** 2  # anti-kT hard jets are close to circular
    rows = []
    for i in range(len(events)):
        ev = events[i]
        stats["events"] += 1
        P = {k: ak.to_numpy(parts[i][k]) for k in ("pt", "eta", "phi", "origin")}

        # Selection uses background-corrected pT. Clean events: rho = 0.
        if args.embed:
            rho_est = rho_grid_median(P["pt"], P["eta"], P["phi"])
            bkg = P["origin"] == 0
            rho_true = float(P["pt"][bkg].sum() / (2 * 3.0 * 2 * np.pi))
        else:
            rho_est = rho_true = 0.0

        jpt = ak.to_numpy(jets[i].pt) if len(jets[i]) else np.empty(0)
        if len(jpt) == 0:
            continue
        jpt_sub = jpt - rho_est * area
        order = np.argsort(-jpt_sub)
        jeta_all = ak.to_numpy(jets[i].eta)
        kept = [j for j in order
                if jpt_sub[j] > args.jet_pt_min and abs(jeta_all[j]) < args.jet_eta_max][: args.max_jets]
        if not kept:
            continue
        stats["events_with_jet"] += 1

        p_id = ak.to_numpy(ev.parton_id)
        p_eta = ak.to_numpy(ev.parton_eta)
        p_phi = ak.to_numpy(ev.parton_phi)

        for rank, j in enumerate(kept):
            stats["jets"] += 1
            jet = jets[i][j]
            j_pt, j_eta, j_phi, j_m = float(jet.pt), float(jet.eta), float(jet.phi) % (2 * np.pi), float(jet.mass)
            j_pt_sub = float(jpt_sub[j])
            if len(p_id) == 0:
                continue
            dr = np.hypot(p_eta - j_eta, dphi(p_phi, j_phi))
            k = int(np.argmin(dr))
            if dr[k] > args.match_dr:
                continue
            stats["matched"] += 1

            ci = ak.to_numpy(idx[i][j])
            cpt, ceta, cphi, corg = P["pt"][ci], P["eta"][ci], P["phi"][ci], P["origin"][ci]
            obs = jet_observables(cpt, ceta, cphi, j_pt, j_eta, j_phi, j_m)

            row = {
                "file_idx": file_idx,
                "event_id": int(ev.event_id),
                "jet_rank": rank,
                "label": int(p_id[k] == 21),
                "parton_id": int(p_id[k]),
                "parton_pt": float(ev.parton_pt[k]),
                "match_dr": float(dr[k]),
                "pt": j_pt, "pt_sub": j_pt_sub, "eta": j_eta, "phi": j_phi,
                "rho_est": rho_est, "rho_true": rho_true,
                **obs,
                "c_pt": cpt.astype(np.float32),
                "c_deta": (ceta - j_eta).astype(np.float32),
                "c_dphi": dphi(cphi, j_phi).astype(np.float32),
                "c_origin": corg.astype(np.int8),
            }
            if args.embed:
                # Truth bookkeeping: what part of this jet came from the background.
                sig = corg == 1
                row["sig_pt_in_jet"] = float(cpt[sig].sum())
                row["bkg_pt_in_jet"] = float(cpt[~sig].sum())
                row["n_bkg_const"] = int((~sig).sum())
                if sig.sum() > 0:
                    s_obs = jet_observables(cpt[sig], ceta[sig], cphi[sig], cpt[sig].sum(), j_eta, j_phi, np.nan)
                    for key in ("n_const", "girth", "ptd", "e2"):
                        row[f"sig_{key}"] = s_obs[key]
            rows.append(row)
    return rows


def pt_weights(pt, label, lo, hi, nbins):
    """Reweight gluon jets to the quark jet pT spectrum; quark weights = 1.
    Jets outside [lo, hi) get weight 0. Weights are scaled so each class sums
    to its own jet count."""
    bins = np.linspace(lo, hi, nbins + 1)
    hq, _ = np.histogram(pt[label == 0], bins=bins, density=True)
    hg, _ = np.histogram(pt[label == 1], bins=bins, density=True)
    b = np.clip(np.digitize(pt, bins) - 1, 0, nbins - 1)
    inside = (pt >= lo) & (pt < hi)
    w = np.ones_like(pt)
    g = label == 1
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(hg > 0, hq / hg, 0.0)
    w[g] = ratio[b[g]]
    w[~inside] = 0.0
    for c in (0, 1):
        m = (label == c) & (w > 0)
        if m.any():
            w[m] *= m.sum() / w[m].sum()
    return w


def summarize(jets, stats_by_file, embed):
    print("\n=== Phase 1 checks ===")
    for f, s in stats_by_file.items():
        ev, ej, nj, nm = s["events"], s["events_with_jet"], s["jets"], s["matched"]
        print(f"[{f}] events {ev}, with a kept jet {ej} ({ej / max(ev, 1):.1%}), "
              f"jets {nj}, matched {nm} ({nm / max(nj, 1):.1%})")
    lab = ak.to_numpy(jets.label)
    w = ak.to_numpy(jets.weight)
    print(f"jets kept: {len(lab)}  quark {np.sum(lab == 0)}  gluon {np.sum(lab == 1)}")
    print("weighted means (gluon should exceed quark for n_const, girth, mass, e2; quark higher for ptd):")
    cols = ["pt", "mass", "n_const", "girth", "ptd", "e2"]
    if embed:
        cols = ["pt", "pt_sub", "sig_pt_in_jet", "bkg_pt_in_jet", "rho_est", "rho_true",
                "mass", "n_const", "girth", "ptd", "e2", "sig_n_const", "sig_girth"]
    print(f"{'':>16}{'quark':>12}{'gluon':>12}")
    for c in cols:
        x = ak.to_numpy(jets[c])
        q, g = lab == 0, lab == 1
        mq = np.average(x[q], weights=w[q]) if w[q].sum() > 0 else np.nan
        mg = np.average(x[g], weights=w[g]) if w[g].sum() > 0 else np.nan
        print(f"{c:>16}{mq:12.4f}{mg:12.4f}")


def plot_checks(jets, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab = ak.to_numpy(jets.label)
    w = ak.to_numpy(jets.weight)
    cols = [("pt", "jet pT [GeV]", None), ("n_const", "constituents", None), ("girth", "girth", None),
            ("ptd", "pT^D", None), ("mass", "jet mass [GeV]", None), ("e2", "e2 (beta = 1)", None)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (c, name, _) in zip(axes.ravel(), cols):
        x = ak.to_numpy(jets[c]).astype(float)
        hi = np.percentile(x, 99.5)
        if c == "n_const":  # integer counts need integer-edged bins
            bins = np.arange(x.min() - 0.5, hi + 1.5, max(1, int((hi - x.min()) // 40) + 1))
        else:
            bins = np.linspace(x.min(), hi, 40)
        for cls, nm in ((0, "quark"), (1, "gluon")):
            m = lab == cls
            ww = None if c == "pt" else w[m]
            ax.hist(x[m], bins=bins, weights=ww, density=True, histtype="step", lw=1.5, label=nm)
        ax.set_xlabel(name)
        ax.legend()
    axes[0, 0].set_title("unweighted (shows the q/g pT difference)", fontsize=9)
    fig.suptitle("Quark vs gluon jets (pT-reweighted except first panel)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="event parquet files from generate_samples.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--R", type=float, default=0.4)
    ap.add_argument("--jet-pt-min", type=float, default=100.0)
    ap.add_argument("--jet-eta-max", type=float, default=2.5)
    ap.add_argument("--const-pt-min", type=float, default=0.3)
    ap.add_argument("--match-dr", type=float, default=0.4)
    ap.add_argument("--max-jets", type=int, default=1, help="1 for Z+jet, 2 for dijets")
    ap.add_argument("--max-events", type=int, default=None, help="per input file")
    ap.add_argument("--chunk", type=int, default=None, help="events per chunk (default 5000 clean, 500 embedded)")
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--bkg-seed", type=int, default=1000)
    ap.add_argument("--n-mean", type=int, default=15000)
    ap.add_argument("--n-sigma", type=float, default=None)
    ap.add_argument("--mean-pt", type=float, default=1.2)
    ap.add_argument("--v2", type=float, default=0.05)
    ap.add_argument("--v3", type=float, default=0.02)
    ap.add_argument("--w-pt-range", type=float, nargs=2, default=[100.0, 400.0])
    ap.add_argument("--w-bins", type=int, default=30)
    ap.add_argument("--plot", default=None, help="save a q/g check figure to this path")
    args = ap.parse_args()

    chunk = args.chunk or (500 if args.embed else 5000)
    bkg_kwargs = {"n_mean": args.n_mean, "n_sigma": args.n_sigma, "mean_pt": args.mean_pt,
                  "vn": {2: args.v2, 3: args.v3}}

    all_rows, stats_by_file = [], {}
    for fi, path in enumerate(args.inputs):
        events = ak.from_parquet(path)
        if args.max_events:
            events = events[: args.max_events]
        stats = {"events": 0, "events_with_jet": 0, "jets": 0, "matched": 0}
        t0 = time.time()
        for s in range(0, len(events), chunk):
            all_rows += process_chunk(events[s: s + chunk], args, fi, bkg_kwargs, stats)
            print(f"  {path}: {min(s + chunk, len(events))}/{len(events)} events "
                  f"({time.time() - t0:.0f} s)", flush=True)
        stats_by_file[path] = stats

    if not all_rows:
        raise SystemExit("no jets passed the cuts")
    jets = ak.Array(all_rows)
    w = pt_weights(ak.to_numpy(jets.pt_sub), ak.to_numpy(jets.label), *args.w_pt_range, args.w_bins)
    jets = ak.with_field(jets, w, "weight")

    meta = {k: v for k, v in vars(args).items() if k not in ("inputs", "plot")}
    meta["inputs"] = list(args.inputs)
    meta["bkg_stream_step"] = BKG_STREAM_STEP
    ak.to_parquet(jets, args.out, parquet_compliant_nested=False, extensionarray=False)
    with open(args.out + ".meta.txt", "w") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")
    print(f"\nwrote {len(jets)} jets to {args.out} (+ {args.out}.meta.txt)")

    summarize(jets, stats_by_file, args.embed)
    if args.plot:
        plot_checks(jets, args.plot)


if __name__ == "__main__":
    main()
