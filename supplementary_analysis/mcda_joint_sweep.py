"""Joint weight x threshold robustness sweep (manuscript Section 5.2).

For each weather case and each upper threshold Tu in {23,24,25,26} C:
  rebuild the four-objective problem with D_Tu, recompute the EXACT Pareto set,
  apply the paper's Eq. 8 normalisation over that set, and run the same 10,000
  uniform weight draws (seed 42, lowest-sim-id tie-break).
Joint first-rank acceptability = average over the four thresholds (uniform Tu prior).
Cross-checks: per-threshold Pareto counts must equal Table row values
(present 247/242/212/211; mid 277/258/262/254; late 265/267/270/288).
Writes mcda_joint_outputs.csv next to this script.
"""
import csv
import os

import numpy as np

# Data resolution: env override first, else paths inside the S1 archive this
# script ships in (../results/...). No machine-specific fallbacks.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _data_dir(env_key, sub):
    for p in (os.environ.get(env_key),
              os.path.join(_HERE, os.pardir, "results", sub)):
        if p and os.path.isdir(p):
            return p if p.endswith(os.sep) else p + os.sep
    raise FileNotFoundError(f"{sub} data not found; set {env_key}")


def _figdir():
    d = os.environ.get("FIGDIR", os.path.join(os.getcwd(), "figures_out"))
    os.makedirs(d, exist_ok=True)
    return d if d.endswith(os.sep) else d + os.sep

ALLC = _data_dir("S1_EXHAUSTIVE", "exhaustive_analysis") + "all_configurations.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcda_joint_outputs.csv")

EXPECTED_PARETO = {
    "2020": {23: 247, 24: 242, 25: 212, 26: 211},
    "2050": {23: 277, 24: 258, 25: 262, 26: 254},
    "2100": {23: 265, 24: 267, 25: 270, 26: 288},
}
HZLABEL = {"2020": "Present", "2050": "Mid-century", "2100": "Late-century"}
THRESHOLDS = [23, 24, 25, 26]

rows = list(csv.DictReader(open(ALLC)))
rng = np.random.default_rng(42)
W = rng.dirichlet(np.ones(4), size=10000)

out_rows = []
for hz in ["2020", "2050", "2100"]:
    sub = [r for r in rows if r["horizon"] == hz]
    assert len(sub) == 625, len(sub)
    ids = np.array([int(r["simulation_id"]) for r in sub])
    base = np.array(
        [
            [float(r["annual_heating_energy_gj"]), float(r["cost_rate_sum_proxy"]),
             float(r["carbon_rate_sum_proxy"])]
            for r in sub
        ]
    )
    D = {tu: np.array([float(r[f"days_below_{tu}_c"]) for r in sub]) for tu in THRESHOLDS}

    joint = {}  # sid -> total wins across thresholds
    per_thr_top = {}
    for tu in THRESHOLDS:
        F = np.column_stack([base, -D[tu]])
        # exact Pareto set over all 625
        leq = (F[None, :, :] <= F[:, None, :]).all(axis=2)
        lt = (F[None, :, :] < F[:, None, :]).any(axis=2)
        dominated = (leq & lt).any(axis=1)
        P = ~dominated
        npar = int(P.sum())
        assert npar == EXPECTED_PARETO[hz][tu], (hz, tu, npar, EXPECTED_PARETO[hz][tu])
        Fp, idp = F[P], ids[P]
        fmin, fmax = Fp.min(0), Fp.max(0)
        Fn = (Fp - fmin) / np.where(fmax > fmin, fmax - fmin, 1.0)
        order = np.argsort(idp)
        win = idp[order[np.argmin((W @ Fn.T)[:, order], axis=1)]]
        wid, cnt = np.unique(win, return_counts=True)
        per_thr_top[tu] = sorted(zip(cnt, wid), reverse=True)[:3]
        for s, c in zip(wid, cnt):
            joint[s] = joint.get(s, 0) + c

    freq = sorted(((c / 400.0, s) for s, c in joint.items()), reverse=True)  # % of 40k
    cum = np.cumsum([f for f, _ in freq])
    k90 = int(np.searchsorted(cum, 90.0)) + 1
    print(f"\n=== {HZLABEL[hz]} ===")
    print(f"  per-threshold Pareto counts OK: {[EXPECTED_PARETO[hz][t] for t in THRESHOLDS]}")
    print(f"  per-threshold top-3: " + " | ".join(
        f"Tu={t}: {[(int(s), round(c/100,1)) for c, s in per_thr_top[t]]}" for t in THRESHOLDS))
    print(f"  JOINT: {len(freq)} distinct winners; {k90} cover 90%")
    print(f"  JOINT top-6: {[(int(s), round(f,1)) for f, s in freq[:6]]}")
    for f, s in freq:
        out_rows.append({"horizon": hz, "simulation_id": int(s), "joint_acceptability_pct": f"{f:.3f}"})

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["horizon", "simulation_id", "joint_acceptability_pct"])
    w.writeheader()
    w.writerows(out_rows)
print(f"\nwrote {OUT}")
