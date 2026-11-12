"""
benchmark.py

Compare background subtraction methods on embedded quark/gluon jets, against
truth jets clustered from the signal particles alone.

Methods
  raw   : no subtraction
  rhoA  : jet pT - rho(phi_jet) * pi R^2 (substructure left uncorrected)
  CS    : event-wide constituent subtraction (C++)
  WAV   : starlet wavelet denoising (C++)

Reported per method, for jets matched to a truth jet (dR < 0.2):
  pT, mass, girth residuals (median and robust width = IQR / 1.349),
  match efficiency, and quark/gluon AUC using girth alone. The truth-jet AUC
  is the ceiling; no method can exceed it with this one variable.
Also reported: the random-cone delta-pT width, i.e. the background
fluctuation that rhoA cannot remove.

Usage:
    python benchmark.py --gen-dir ../gen --events 600
"""
import argparse
import sys
import time

import awkward as ak
import fastjet
import numpy as np
import vector

vector.register_awkward()
R = 0.4
CONST_MIN = 0.3


def dphi(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def obs(pt, eta, phi, jeta, jphi, jpt):
    dr = np.hypot(eta - jeta, dphi(phi, jphi))
    E = pt * np.cosh(eta)
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    m2 = E.sum() ** 2 - px.sum() ** 2 - py.sum() ** 2 - pz.sum() ** 2
    return {"pt": jpt, "mass": float(np.sqrt(max(m2, 0.0))), "girth": float((pt * dr).sum() / jpt)}


def cluster(events):
    """events: list of (pt, eta, phi) numpy triples. Returns per-event list of
    jets as dicts with axis and constituent arrays."""
    arr = ak.zip({"pt": ak.Array([e[0] for e in events]), "eta": ak.Array([e[1] for e in events]),
                  "phi": ak.Array([e[2] for e in events]),
                  "mass": ak.Array([np.zeros_like(e[0]) for e in events])}, with_name="Momentum4D")
    cs = fastjet.ClusterSequence(arr, fastjet.JetDefinition(fastjet.antikt_algorithm, R))
    jets = cs.inclusive_jets(min_pt=20.0)
    idx = cs.constituent_index(min_pt=20.0)
    out = []
    for i, (p, e, f) in enumerate(events):
        js = []
        for j in range(len(jets[i])):
            ci = ak.to_numpy(idx[i][j])
            js.append({"pt": float(jets[i][j].pt), "eta": float(jets[i][j].eta),
                       "phi": float(jets[i][j].phi) % (2 * np.pi), "c": (p[ci], e[ci], f[ci])})
        out.append(js)
    return out


def nearest(jets, eta, phi, maxdr=0.2):
    best, bd = None, maxdr
    for j in jets:
        d = np.hypot(j["eta"] - eta, dphi(j["phi"], phi))
        if d < bd:
            best, bd = j, d
    return best


def auc(score, label):
    """Rank-based ROC AUC (gluon = positive)."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    npos, nneg = label.sum(), (1 - label).sum()
    return (ranks[label == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def width(x):
    q1, q3 = np.percentile(x, [25, 75])
    return (q3 - q1) / 1.349


PT_EDGES = list(np.geomspace(CONST_MIN, 60.0, 21))
R_EDGES = list(np.linspace(0.0, 0.5, 11))


def build_template(gen_dir, start, stop):
    """Signal density s(pT, r) per unit area per pT bin around truth jet axes,
    averaged over jets, from clean Pythia events [start, stop) of both files."""
    H = np.zeros((len(PT_EDGES) - 1, len(R_EDGES) - 1))
    n_jets = 0
    for fname in ("q3k.parquet", "g3k.parquet"):
        ev = ak.from_parquet(f"{gen_dir}/{fname}")[start:stop]
        sig = []
        for e in ev:
            p, h, f = (ak.to_numpy(e.pt).astype(float), ak.to_numpy(e.eta).astype(float),
                       ak.to_numpy(e.phi).astype(float))
            k = p > CONST_MIN
            sig.append((p[k], h[k], f[k]))
        for (p, h, f), js in zip(sig, cluster(sig)):
            js = [j for j in js if j["pt"] > 100 and abs(j["eta"]) < 2.5]
            if not js:
                continue
            j = max(js, key=lambda x: x["pt"])
            r = np.hypot(h - j["eta"], dphi(f, j["phi"]))
            H += np.histogram2d(p, r, bins=[PT_EDGES, R_EDGES])[0]
            n_jets += 1
    ring = np.pi * (np.array(R_EDGES[1:]) ** 2 - np.array(R_EDGES[:-1]) ** 2)
    return H / n_jets / ring[None, :], n_jets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default="../gen")
    ap.add_argument("--events", type=int, default=600, help="per class")
    ap.add_argument("--k", type=float, nargs="+", default=[2.0, 3.0], help="wavelet thresholds")
    ap.add_argument("--carry", type=int, nargs="+", default=[3], help="wavelet carry levels (0 = clip)")
    ap.add_argument("--cs-dr", type=float, nargs="+", default=[0.25])
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--train-start", type=int, default=2000, help="template events (not tested on)")
    args = ap.parse_args()

    sys.path.insert(0, args.gen_dir)
    import generate_samples as gs
    import jetdenoise as jd

    rng = np.random.default_rng(args.seed)
    methods = (["raw", "rhoA"] + [f"CS dR{d}" for d in args.cs_dr]
               + [f"WAV k{k:g}c{c}" for k in args.k for c in args.carry] + ["POST"])
    assert args.events <= args.train_start, "test events would overlap the template events"
    tmpl, n_tmpl = build_template(args.gen_dir, args.train_start, 3000)
    print(f"signal template from {n_tmpl} training jets")
    results = {m: [] for m in methods}
    truth_rows, dpt, v2_err = [], [], []
    t_cpp = {m: 0.0 for m in methods}

    for label, fname, fi in ((0, "q3k.parquet", 0), (1, "g3k.parquet", 1)):
        ev = ak.from_parquet(f"{args.gen_dir}/{fname}")[: args.events]
        sig_events, mix_events, meta = [], [], []
        for e in ev:
            p, h, f = (ak.to_numpy(e.pt).astype(float), ak.to_numpy(e.eta).astype(float),
                       ak.to_numpy(e.phi).astype(float))
            keep = p > CONST_MIN
            sig = (p[keep], h[keep], f[keep])
            b, tr = gs.thermal_background(args.seed + 7919 * fi, int(e.event_id))
            bp, bh, bf = b[:, 0].astype(float), b[:, 1].astype(float), b[:, 2].astype(float)
            kb = bp > CONST_MIN
            mix = (np.concatenate([sig[0], bp[kb]]), np.concatenate([sig[1], bh[kb]]),
                   np.concatenate([sig[2], bf[kb]]))
            sig_events.append(sig)
            mix_events.append(mix)
            meta.append((e, tr))

        truth_jets = cluster(sig_events)
        per_method_events = {m: [] for m in methods}
        bmaps = []
        for (mp, mh, mf), (e, tr) in zip(mix_events, meta):
            t0 = time.perf_counter()
            bm = jd.background_map(mp, mh, mf)
            t_bm = time.perf_counter() - t0
            bmaps.append(bm)
            v2_err.append(bm.vn(2) - tr["vn"][2])
            per_method_events["raw"].append((mp, mh, mf))
            per_method_events["rhoA"].append((mp, mh, mf))
            for d in args.cs_dr:
                t0 = time.perf_counter()
                cp = jd.constituent_subtract(mp, mh, mf, bm, max_dr=d)
                t_cpp[f"CS dR{d}"] += time.perf_counter() - t0 + t_bm
                k = cp > 1e-9
                per_method_events[f"CS dR{d}"].append((cp[k], mh[k], mf[k]))
            for kk in args.k:
                for cc in args.carry:
                    name = f"WAV k{kk:g}c{cc}"
                    t0 = time.perf_counter()
                    w = jd.starlet_denoise(mp, mh, mf, bm, k_sigma=kk, carry_levels=cc)["weight"]
                    t_cpp[name] += time.perf_counter() - t0 + t_bm
                    wp = mp * w
                    k = wp > 1e-9
                    per_method_events[name].append((wp[k], mh[k], mf[k]))

        clustered = {m: cluster(per_method_events[m]) for m in methods if m != "POST"}
        cs_name = f"CS dR{args.cs_dr[0]}"
        post_events = []
        for i, (mp, mh, mf) in enumerate(mix_events):
            t0 = time.perf_counter()
            axes = [j for j in clustered[cs_name][i] if j["pt"] > 50 and abs(j["eta"]) < 2.8]
            ae, af = [j["eta"] for j in axes], [j["phi"] for j in axes]
            bs = jd.background_spectrum(mp, mh, mf, ae, af, PT_EDGES, bmaps[i])
            w = jd.posterior_weights(mp, mh, mf, ae, af, PT_EDGES, R_EDGES, tmpl, list(bs), bmaps[i])
            t_cpp["POST"] += time.perf_counter() - t0
            wp = mp * w
            k = wp > 1e-9
            post_events.append((wp[k], mh[k], mf[k]))
        clustered["POST"] = cluster(post_events)

        for i, (e, tr) in enumerate(meta):
            # truth jet = leading signal jet, pT > 100, |eta| < 2.5, matched to the parton
            tj = [j for j in truth_jets[i] if j["pt"] > 100 and abs(j["eta"]) < 2.5]
            if not tj:
                continue
            tj = max(tj, key=lambda j: j["pt"])
            pe, pf = ak.to_numpy(e.parton_eta), ak.to_numpy(e.parton_phi)
            if len(pe) == 0 or np.min(np.hypot(pe - tj["eta"], dphi(pf, tj["phi"]))) > 0.4:
                continue
            t_obs = obs(*tj["c"], tj["eta"], tj["phi"], tj["pt"])
            truth_rows.append({**t_obs, "label": label})
            n_truth = len(truth_rows) - 1
            bm = bmaps[i]
            for m in methods:
                rj = nearest(clustered[m][i], tj["eta"], tj["phi"])
                if rj is None:
                    results[m].append(None)
                    continue
                o = obs(*rj["c"], rj["eta"], rj["phi"], rj["pt"])
                if m == "rhoA":
                    o["pt"] = rj["pt"] - bm.rho(rj["phi"]) * np.pi * R ** 2
                results[m].append(o)
            # random cone, kept at least 1.0 away from the truth jet
            mp, mh, mf = mix_events[i]
            bonly = (mp[len(sig_events[i][0]):], mh[len(sig_events[i][0]):], mf[len(sig_events[i][0]):])
            for _ in range(5):
                ce, cf = rng.uniform(-2.5, 2.5), rng.uniform(0, 2 * np.pi)
                if np.hypot(ce - tj["eta"], dphi(cf, tj["phi"])) < 1.0:
                    continue
                inc = np.hypot(bonly[1] - ce, dphi(bonly[2], cf)) < R
                dpt.append(bonly[0][inc].sum() - bm.rho(cf) * np.pi * R ** 2)
            _ = n_truth

    T = {k: np.array([r[k] for r in truth_rows]) for k in ("pt", "mass", "girth", "label")}
    lab = T["label"].astype(int)
    n = len(lab)
    print(f"\ntruth jets: {n} (quark {np.sum(lab == 0)}, gluon {np.sum(lab == 1)})")
    print(f"random-cone delta-pT: mean {np.mean(dpt):+.1f} GeV, width {np.std(dpt):.1f} GeV "
          f"(robust {width(np.array(dpt)):.1f})")
    print(f"v2 estimate minus truth: mean {np.mean(v2_err):+.4f}, spread {np.std(v2_err):.4f}")
    print(f"girth q/g AUC on truth jets (ceiling): {auc(T['girth'], lab):.3f}\n")

    hdr = (f"{'method':<13}{'eff':>6}{'dpT med':>9}{'dpT wid':>9}{'dm med':>8}{'dm wid':>8}"
           f"{'dg med':>9}{'dg wid':>9}{'AUC':>7}{'ms/evt':>8}")
    print(hdr)
    print("-" * len(hdr))
    n_evt = 2 * args.events
    summary = {}
    for m in methods:
        ok = np.array([r is not None for r in results[m]])
        R_ = {k: np.array([r[k] for r in results[m] if r is not None]) for k in ("pt", "mass", "girth")}
        d = {k: R_[k] - T[k][ok] for k in R_}
        a = auc(R_["girth"], lab[ok])
        ms = 1e3 * t_cpp[m] / n_evt
        summary[m] = (ok.mean(), np.median(d["pt"]), width(d["pt"]), a)
        print(f"{m:<13}{ok.mean():6.3f}{np.median(d['pt']):+9.1f}{width(d['pt']):9.1f}"
              f"{np.median(d['mass']):+8.1f}{width(d['mass']):8.1f}"
              f"{np.median(d['girth']):+9.4f}{width(d['girth']):9.4f}{a:7.3f}"
              f"{(f'{ms:.1f}' if ms else '-'):>8}")
    print("\ndpT, dm in GeV; dg = girth residual. wid = IQR/1.349. AUC uses girth only.")
    print("ms/evt: C++ time per event; POST excludes the CS pass and clustering it reuses for jet axes.")


if __name__ == "__main__":
    main()
