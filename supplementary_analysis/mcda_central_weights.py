"""SMAA first-rank acceptability and central weight vectors on the exact Pareto fronts.

For each weather case, 1,000,000 uniform Dirichlet(1,1,1,1) weight draws (seed 42,
chunked) are scored on the exact front under the paper's Eq. (6) normalisation and
Eq. (7) weighted sum with the sweep tie rule (exact-equality argmin over id-sorted
scores, lowest simulation id first). For every package that wins at least one draw:
  - first-rank acceptability a_i = share of draws it wins (Lahdelma et al., 1998);
  - central weight vector w_i^c = mean of the winning draws, the expected centre of
    gravity of the package's favourable weight set under a uniform weight prior.
Both quantities come from the same 1,000,000-draw run, so the table is
self-consistent; the 10,000-draw sweep of mcda_supportedness_sweep.py is kept
unchanged for the published figures, and this script prints the comparison so the
rounding-level agreement between the two sample sizes can be stated. 9,604 draws
give +/-0.01 accuracy at 95% confidence for first-rank shares (Tervonen & Figueira,
2008); one million draws stabilise the central vectors of ~1%-acceptability packages.

No sampled min/max winning weights are reported: sample extrema are not the
theoretical weight bounds of the favourable set.

Writes mcda_central_weights.csv. Deterministic: fixed seed 42.
"""
import csv
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _data_dir(env_key, sub):
    for p in (os.environ.get(env_key), os.path.join(_HERE, os.pardir, "results", sub)):
        if p and os.path.isdir(p):
            return p if p.endswith(os.sep) else p + os.sep
    raise FileNotFoundError(f"{sub} data not found; set {env_key}")


BASE = _data_dir("S1_EXHAUSTIVE", "exhaustive_analysis")
OUTCSV = os.path.join(_HERE, "mcda_central_weights.csv")

HORIZONS = ["2020", "2050", "2100"]
HZLABEL = {"2020": "Present", "2050": "Mid-century", "2100": "Late-century"}
N_DRAWS = 1_000_000
CHUNK = 50_000

rows = list(csv.DictReader(open(os.path.join(BASE, "pareto_fronts.csv"))))

fronts = {}
for hz in HORIZONS:
    sub = [r for r in rows if r["horizon"] == hz]
    ids = np.array([int(r["simulation_id"]) for r in sub])
    F = np.array(
        [
            [
                float(r["annual_heating_energy_gj"]),
                float(r["cost_rate_sum_proxy"]),
                float(r["carbon_rate_sum_proxy"]),
                -float(r["days_below_24_c"]),
            ]
            for r in sub
        ]
    )
    fmin, fmax = F.min(0), F.max(0)
    Fn = (F - fmin) / np.where(fmax > fmin, fmax - fmin, 1.0)
    order = np.argsort(ids)
    fronts[hz] = (ids[order], Fn[order])  # id-sorted so argmin ties break to lowest id

rng = np.random.default_rng(42)
win_count = {hz: np.zeros(len(fronts[hz][0]), dtype=np.int64) for hz in HORIZONS}
weight_sum = {hz: np.zeros((len(fronts[hz][0]), 4)) for hz in HORIZONS}

done = 0
while done < N_DRAWS:
    m = min(CHUNK, N_DRAWS - done)
    W = rng.dirichlet(np.ones(4), size=m)
    for hz in HORIZONS:
        ids_o, Fn_o = fronts[hz]
        k = np.argmin(W @ Fn_o.T, axis=1)
        np.add.at(win_count[hz], k, 1)
        np.add.at(weight_sum[hz], k, W)
    done += m
print(f"{done:,} draws scored per weather case")

# 10,000-draw comparison sample (same seed and generator pattern as the sweep script)
W10 = np.random.default_rng(42).dirichlet(np.ones(4), size=10000)

out = []
for hz in HORIZONS:
    ids_o, Fn_o = fronts[hz]
    k10 = np.argmin(W10 @ Fn_o.T, axis=1)
    c10 = np.bincount(k10, minlength=len(ids_o))
    a1m = win_count[hz] / N_DRAWS * 100.0
    top = np.argsort(-a1m)[:5]
    print(f"\n=== {HZLABEL[hz]} ({hz}) — top 5 by first-rank acceptability (1M draws)")
    for j in top:
        wc = weight_sum[hz][j] / win_count[hz][j]
        print(f"  package {ids_o[j]:>3d}  a_i {a1m[j]:6.2f}%  (10k: {c10[j] / 100.0:5.2f}%)  "
              f"w^c = ({wc[0]:.3f}, {wc[1]:.3f}, {wc[2]:.3f}, {wc[3]:.3f})")
    for j in np.where(win_count[hz] > 0)[0]:
        wc = weight_sum[hz][j] / win_count[hz][j]
        out.append({
            "horizon": hz,
            "simulation_id": int(ids_o[j]),
            "acceptability_pct_1m": f"{a1m[j]:.3f}",
            "acceptability_pct_10k": f"{c10[j] / 100.0:.2f}",
            "wc_energy": f"{wc[0]:.4f}",
            "wc_cost": f"{wc[1]:.4f}",
            "wc_gwp": f"{wc[2]:.4f}",
            "wc_d24": f"{wc[3]:.4f}",
        })

with open(OUTCSV, "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    wtr.writeheader()
    wtr.writerows(out)
print(f"\nwrote {OUTCSV} ({len(out)} rows)")
