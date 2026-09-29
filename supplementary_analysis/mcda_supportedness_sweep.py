"""Unique-selectability LP test, uniform weight sweep, and ASF comparison on the exact Pareto fronts.

Reproduces the Section 3.4 MCDA results of the manuscript from the released S1 data:
  (1) exact LP unique-selectability test for every Pareto member: maximise the
      worst-case weighted-sum score margin t* over admissible weights (weight
      floor 1e-6). A member is uniquely_selectable when t* > 1e-9 and
      not_uniquely_selectable otherwise; no boundary/unsupported split is
      certified, because several margins sit at solver-tolerance scale;
  (2) 10,000-draw uniform (Dirichlet(1,1,1,1)) weight sweep: first-rank selection
      frequency under the paper's Eq. (6) normalisation and lowest-sim-id tie-break;
  (3) achievement-scalarising-function (ASF; Wierzbicki, 1980) selections for the
      five weight profiles, compared with the weighted-sum (Eq. 7) selections,
      with a rho sensitivity grid {1e-8, 1e-6, 1e-4, 1e-2};
  (4) normalisation-anchor check: the 15 profile selections recomputed with the
      Eq. (6) scaling constants taken over all 625 packages instead of the front;
  (5) warm-side baseline safeguard: the 15 profile selections recomputed after
      requiring D24 >= the same-case baseline count before scoring, reported both
      with the front's original normalisation (feasibility screen) and re-normalised
      over the screened subset.
Writes mcda_outputs.csv and mcda_profile_sensitivity.csv (per-case per-profile
weighted-sum and ASF selections, their agreement, and the rho-grid stability flag,
so the supplement generator can derive the ASF-comparison claims from released
data). The coverage curves formerly shown by fig_weight_sweep are carried by
make_decision_robustness_figure.py.

Tie conventions (two deliberate, distinct rules):
  - the five named-profile selections use a numerical tie rule
    np.isclose(..., rtol=0.0, atol=1e-12) followed by the lowest simulation id.
    rtol MUST be 0.0: numpy's default relative tolerance (1e-05) treats genuinely
    unequal scores as tied and changes two ASF selections;
  - the 10,000-draw sweep uses exact-equality argmin over id-sorted scores
    (deterministic, lowest id first). A tolerance rule there could manufacture
    ties between genuinely unequal scores and is intentionally not used.

Usage: point BASE at the extracted S1 results/exhaustive_analysis directory and run.
Deterministic: fixed seed 42; LP via scipy HiGHS.
"""
import csv
import os

import numpy as np
from scipy.optimize import linprog

# Data resolution: env override first, else paths inside the S1 archive this
# script ships in (../results/...). No machine-specific fallbacks.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _data_dir(env_key, sub):
    for p in (os.environ.get(env_key),
              os.path.join(_HERE, os.pardir, "results", sub)):
        if p and os.path.isdir(p):
            return p if p.endswith(os.sep) else p + os.sep
    raise FileNotFoundError(f"{sub} data not found; set {env_key}")


BASE = _data_dir("S1_EXHAUSTIVE", "exhaustive_analysis")
OUTCSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcda_outputs.csv")
PROFCSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcda_profile_sensitivity.csv")

rows = list(csv.DictReader(open(os.path.join(BASE, "pareto_fronts.csv"))))
all_rows = list(csv.DictReader(open(os.path.join(BASE, "all_configurations.csv"))))
HORIZONS = ["2020", "2050", "2100"]
HZLABEL = {"2020": "Present", "2050": "Mid-century", "2100": "Late-century"}


def objective_matrix(sub):
    return np.array(
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

PROFILES = {
    "Cost": np.array([0.1, 0.7, 0.1, 0.1]),
    "GWP": np.array([0.1, 0.1, 0.7, 0.1]),
    "Energy": np.array([0.7, 0.1, 0.1, 0.1]),
    "D24": np.array([0.1, 0.1, 0.1, 0.7]),
    "Balanced": np.array([0.25, 0.25, 0.25, 0.25]),
}
RHO = 1e-6  # ASF augmentation parameter (headline value)
RHO_GRID = [1e-8, 1e-6, 1e-4, 1e-2]  # sensitivity grid; selections must not move

rng = np.random.default_rng(42)
W = rng.dirichlet(np.ones(4), size=10000)

out_rows = []
profile_rows = []
asf_summary = {}

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
    n = len(sub)
    fmin, fmax = F.min(0), F.max(0)
    span = np.where(fmax > fmin, fmax - fmin, 1.0)
    Fn = (F - fmin) / span  # Eq. (6) normalisation; minimise
    Y = -Fn

    def select(scores):
        best = np.min(scores)
        # rtol=0.0 is load-bearing: the numpy default (1e-05) manufactures ties
        # between unequal scores and changes two ASF selections.
        cand = np.where(np.isclose(scores, best, rtol=0.0, atol=1e-12))[0]
        return cand[np.argmin(ids[cand])]

    # --- weighted-sum (Eq. 7) and ASF (Wierzbicki 1980, ideal point = 0) selections
    ws_sel, asf_sel = {}, {}
    for name, w in PROFILES.items():
        ws_sel[name] = ids[select(Fn @ w)]
        asf_scores = np.max(w * Fn, axis=1) + RHO * (Fn @ w)
        asf_sel[name] = ids[select(asf_scores)]
    agree = sum(ws_sel[p] == asf_sel[p] for p in PROFILES)
    asf_summary[hz] = (ws_sel, asf_sel, agree)
    print(f"\n=== {hz} | n={n}")
    print("  weighted-sum:", ws_sel)
    print("  ASF        :", asf_sel, f"| agreement {agree}/5")

    # --- ASF rho sensitivity: selections must be identical across the grid
    for rho in RHO_GRID:
        sel_rho = {name: ids[select(np.max(w * Fn, axis=1) + rho * (Fn @ w))]
                   for name, w in PROFILES.items()}
        assert sel_rho == asf_sel, (hz, rho, sel_rho, asf_sel)
    print(f"  ASF selections stable for rho in {RHO_GRID}")
    for name in PROFILES:
        profile_rows.append({
            "horizon": hz,
            "profile": name,
            "weighted_sum_selection": int(ws_sel[name]),
            "asf_selection": int(asf_sel[name]),
            "agreement": bool(ws_sel[name] == asf_sel[name]),
            "asf_stable_over_rho_grid": True,
        })

    # --- LP unique-selectability test
    status, tstars = {}, {}
    for k in range(n):
        diff = np.delete(Y[k] - Y, k, axis=0)
        c = np.array([0, 0, 0, 0, -1.0])
        A_ub = np.hstack([-diff, np.ones((n - 1, 1))])
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=np.zeros(n - 1),
            A_eq=np.array([[1, 1, 1, 1, 0.0]]),
            b_eq=np.array([1.0]),
            bounds=[(1e-6, None)] * 4 + [(None, None)],
            method="highs",
        )
        assert res.status == 0, (hz, int(ids[k]), res.status, res.message)
        t = -res.fun
        tstars[ids[k]] = t
        status[ids[k]] = "uniquely_selectable" if t > 1e-9 else "not_uniquely_selectable"
    nu = sum(1 for v in status.values() if v == "not_uniquely_selectable")
    margins = [-tstars[i] for i, v in status.items() if v == "not_uniquely_selectable"]
    print(
        f"  uniquely selectable {n - nu}/{n}; not uniquely selectable {nu}/{n} "
        f"({100 * nu / n:.0f}%); -t* median {np.median(margins):.3e} max {max(margins):.3e}"
    )
    for name, sid in ws_sel.items():
        assert status[sid] == "uniquely_selectable", f"{name} selection not uniquely selectable?!"

    # --- uniform weight sweep (exact-equality argmin tie rule; see module docstring)
    scores = W @ Fn.T
    order = np.argsort(ids)
    winners = ids[order[np.argmin(scores[:, order], axis=1)]]
    win_ids, counts = np.unique(winners, return_counts=True)
    freq = sorted(zip(counts / 100.0, win_ids), reverse=True)
    cum = np.cumsum([f for f, _ in freq])
    k90 = int(np.searchsorted(cum, 90.0)) + 1
    nus_wins = sum(c for c, i in zip(counts, win_ids) if status[i] == "not_uniquely_selectable")
    print(
        f"  sweep: {len(win_ids)} members ever selected; {k90} cover 90%; "
        f"not-uniquely-selectable wins {nus_wins}; top-3 {[ (i, round(f,1)) for f, i in freq[:3] ]}"
    )
    # --- (4) normalisation-anchor check: Eq. (6) constants over all 625 packages
    sub625 = [r for r in all_rows if r["horizon"] == hz]
    assert len(sub625) == 625, len(sub625)
    F625 = objective_matrix(sub625)
    lo, hi = F625.min(0), F625.max(0)
    Fn_alt = (F - lo) / np.where(hi > lo, hi - lo, 1.0)
    alt_sel = {name: ids[select(Fn_alt @ w)] for name, w in PROFILES.items()}
    changed = {p for p in PROFILES if alt_sel[p] != ws_sel[p]}
    print(f"  all-625 anchors: selections {'unchanged' if not changed else 'CHANGED ' + str(sorted(changed))}")

    # --- (5) warm-side baseline safeguard: D24 >= same-case baseline before scoring
    d24 = -F[:, 3]
    d24_base = float(d24[ids == 1][0])
    feas = d24 >= d24_base
    scr_fixed, scr_renorm = {}, {}
    Ff = F[feas]
    fmin_s, fmax_s = Ff.min(0), Ff.max(0)
    Fn_scr = (Ff - fmin_s) / np.where(fmax_s > fmin_s, fmax_s - fmin_s, 1.0)
    ids_f = ids[feas]

    def select_among(scores, sel_ids):
        best = np.min(scores)
        cand = np.where(np.isclose(scores, best, rtol=0.0, atol=1e-12))[0]
        return sel_ids[cand[np.argmin(sel_ids[cand])]]

    for name, w in PROFILES.items():
        scr_fixed[name] = select_among((Fn[feas]) @ w, ids_f)   # front normalisation kept
        scr_renorm[name] = select_among(Fn_scr @ w, ids_f)       # re-normalised over screen
    for label, sel in (("fixed-norm", scr_fixed), ("re-norm", scr_renorm)):
        moved = {p: (int(ws_sel[p]), int(sel[p])) for p in PROFILES if sel[p] != ws_sel[p]}
        print(f"  D24>=baseline screen ({label}): "
              f"{'no selection changes' if not moved else 'changed ' + str(moved)}")
        for p, (old, new) in moved.items():
            io, in_ = np.where(ids == old)[0][0], np.where(ids == new)[0][0]
            dF = F[in_] - F[io]
            print(f"    {p}: {old} -> {new} | dE_H {dF[0]:+.3f} GJ, dI_C {dF[1]:+.1f}, "
                  f"dI_G {dF[2]:+.1f}, dD24 {-dF[3]:+.0f} d")

    accept = dict(zip(win_ids, counts / 100.0))
    for i in ids:
        out_rows.append(
            {
                "horizon": hz,
                "simulation_id": int(i),
                "selectability": status[i],
                "t_star": f"{tstars[i]:.3e}",
                "acceptability_pct": f"{accept.get(i, 0.0):.2f}",
            }
        )

with open(OUTCSV, "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    wtr.writeheader()
    wtr.writerows(out_rows)
print(f"\nwrote {OUTCSV} ({len(out_rows)} rows)")

with open(PROFCSV, "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(profile_rows[0].keys()))
    wtr.writeheader()
    wtr.writerows(profile_rows)
print(f"wrote {PROFCSV} ({len(profile_rows)} rows)")
