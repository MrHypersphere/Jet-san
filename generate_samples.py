"""
generate_samples.py

Vacuum signal from Pythia 8 plus a reproducible toy thermal background,
for quark/gluon classification and background-subtraction studies.

Install (Linux or WSL2; the pythia8mc wheels are Linux-only):
    pip install -r requirements.txt

Examples:
    # pure-ish quark and gluon samples (Z -> nu nu recoiling against one parton)
    python generate_samples.py --process zq --nevents 100000 --seed 11 --out zq.parquet
    python generate_samples.py --process zg --nevents 100000 --seed 12 --out zg.parquet

    # natural quark/gluon mixture (inclusive dijets)
    python generate_samples.py --process dijet --nevents 100000 --seed 13 --out dijet.parquet

Design choice: only the SIGNAL particles are written to disk. The background
for event i is regenerated deterministically from (bkg_seed, i) with
thermal_background(). A 15k-particle background is ~0.4 MB/event in float32,
so storing it for 100k events would cost ~40 GB. Regenerating it on the fly
also lets you swap backgrounds between events and scale rho without
regenerating anything.
"""
import argparse

import awkward as ak
import numpy as np

NEUTRINOS = {12, 14, 16}
PION_MASS = {211: 0.13957, -211: 0.13957, 111: 0.13498}


def make_pythia(process, ecm=5020.0, pthat_min=80.0, pthat_max=-1.0, seed=1):
    import pythia8mc

    py = pythia8mc.Pythia()
    s = py.readString
    s("Print:quiet = on")
    s("Beams:idA = 2212")
    s("Beams:idB = 2212")
    s(f"Beams:eCM = {ecm}")
    s(f"PhaseSpace:pTHatMin = {pthat_min}")
    if pthat_max > 0:
        s(f"PhaseSpace:pTHatMax = {pthat_max}")
    s("Random:setSeed = on")
    s(f"Random:seed = {seed}")

    if process == "zg":
        s("WeakBosonAndParton:qqbar2gmZg = on")
    elif process == "zq":
        s("WeakBosonAndParton:qg2gmZq = on")
    elif process == "dijet":
        s("HardQCD:all = on")
    elif process == "gammajet":
        s("PromptPhoton:all = on")
    else:
        raise ValueError(f"unknown process {process}")

    if process in ("zg", "zq"):
        s("WeakZ0:gmZmode = 2")
        s("23:onMode = off")
        s("23:onIfAny = 12 14 16")

    if not py.init():
        raise RuntimeError("Pythia initialization failed")
    return py


def hard_partons(ev):
    """Outgoing partons of the hardest process (status -23)."""
    out = []
    for i in range(ev.size()):
        p = ev[i]
        if p.status() == -23 and (p.id() == 21 or 1 <= abs(p.id()) <= 6):
            out.append((p.id(), p.pT(), p.eta(), p.phi() % (2 * np.pi)))
    return out


def final_particles(ev, eta_max):
    """Visible final-state particles within |eta| < eta_max."""
    rows = []
    for i in range(ev.size()):
        p = ev[i]
        if not p.isFinal() or abs(p.id()) in NEUTRINOS:
            continue
        if abs(p.eta()) > eta_max:
            continue
        rows.append((p.pT(), p.eta(), p.phi() % (2 * np.pi), p.m(), p.id(), p.charge(), 1.0))
    return np.asarray(rows, dtype=np.float32).reshape(-1, 7)


def _sample_phi(rng, n, vn, psin):
    """Sample phi from 1 + 2 sum(v_n cos(n(phi - Psi_n)))."""
    fmax = 1.0 + 2.0 * sum(abs(v) for v in vn.values())
    out = np.empty(0)
    while out.size < n:
        m = int(1.3 * (n - out.size)) + 16
        phi = rng.uniform(0.0, 2 * np.pi, size=m)
        f = 1.0 + sum(2 * v * np.cos(k * (phi - psin[k])) for k, v in vn.items())
        out = np.concatenate([out, phi[rng.uniform(0.0, fmax, size=m) < f]])
    return out[:n]


def thermal_background(bkg_seed, event_id, n_mean=15000, n_sigma=None, mean_pt=1.2,
                       eta_max=3.0, vn=None):
    """Return reproducible independent-emission toy background and its truth."""
    rng = np.random.default_rng([bkg_seed, event_id])
    vn = {2: 0.05, 3: 0.02} if vn is None else vn
    n_sigma = np.sqrt(n_mean) if n_sigma is None else n_sigma
    n = max(int(round(rng.normal(n_mean, n_sigma))), 0)
    psin = {k: rng.uniform(0.0, 2 * np.pi / k) for k in vn}
    pt = rng.gamma(shape=2.0, scale=mean_pt / 2.0, size=n)
    eta = rng.uniform(-eta_max, eta_max, size=n)
    phi = _sample_phi(rng, n, vn, psin)
    pid = rng.choice([211, -211, 111], size=n)
    m = np.vectorize(PION_MASS.get)(pid) if n else np.empty(0)
    charge = np.sign(pid) * (np.abs(pid) == 211)
    parts = np.column_stack([pt, eta, phi, m, pid, charge, np.zeros(n)]).astype(np.float32)
    truth = {"n": n, "vn": vn, "psin": psin, "sum_pt": float(pt.sum())}
    return parts, truth


def embed(signal_parts, bkg_seed, event_id, **bkg_kwargs):
    """Return signal plus thermal background, retaining the origin flag."""
    bkg, truth = thermal_background(bkg_seed, event_id, **bkg_kwargs)
    return np.concatenate([signal_parts, bkg]), truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", required=True, choices=["zq", "zg", "dijet", "gammajet"])
    ap.add_argument("--nevents", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bkg-seed", type=int, default=1000)
    ap.add_argument("--pthat-min", type=float, default=80.0)
    ap.add_argument("--eta-max", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    py = make_pythia(args.process, pthat_min=args.pthat_min, seed=args.seed)
    records, n_fail = [], 0
    while len(records) < args.nevents:
        if not py.next():
            n_fail += 1
            if n_fail > 1000:
                raise RuntimeError("too many failed events")
            continue
        parts = final_particles(py.event, args.eta_max)
        partons = hard_partons(py.event)
        records.append({
            "event_id": len(records), "bkg_seed": args.bkg_seed,
            "pt": parts[:, 0], "eta": parts[:, 1], "phi": parts[:, 2],
            "m": parts[:, 3], "pid": parts[:, 4].astype(np.int32),
            "charge": parts[:, 5].astype(np.int8),
            "parton_id": np.array([p[0] for p in partons], dtype=np.int32),
            "parton_pt": np.array([p[1] for p in partons], dtype=np.float32),
            "parton_eta": np.array([p[2] for p in partons], dtype=np.float32),
            "parton_phi": np.array([p[3] for p in partons], dtype=np.float32),
        })

    ak.to_parquet(ak.Array(records), args.out)
    print(f"wrote {len(records)} events to {args.out}")


if __name__ == "__main__":
    main()
