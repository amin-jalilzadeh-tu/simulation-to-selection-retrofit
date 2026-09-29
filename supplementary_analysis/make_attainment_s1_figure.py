"""Supplementary Figure S1.7: attainment by floor state and matched effects.

Panel (a) separates the exact epsilon-constraint attainment envelope into two
package families: ground floor left at state 0 and any insulated ground-floor
state.  For family F,

  g_F(r) = max { D24(x) - D24(baseline) : x in F,
                 E_H(baseline) - E_H(x) >= r }.

The upper family envelope is the unrestricted answer.  Separating the families
shows that the mid- and late-century drops occur when the floor-free family runs
out of heating reduction, not along a continuous physical response.  The
staircases use ``steps-pre`` because a package attaining exactly r remains
feasible at r and is excluded immediately above it.

Each callout names the largest reduction available without floor insulation
and the decrease in the best attainable day count when a strictly greater
reduction is required.  The drops are 2, 39 and 48 days under present,
mid-century and late-century weather, from limits of 17.3, 14.7 and 10.7 GJ/year.
The exact values and post-limit optima are asserted below.

Panel (a) uses one continuous day-count scale.  Only family endpoints carry
markers; intermediate points on a constraint envelope need not represent
individual package outcomes.

Panels (b)-(c) retain matched effects from the complete factorial design.  For
each component, its thermally strongest state is compared with state 0 while the
other three components are held fixed, giving 125 deterministic contexts per
component and weather case.  For windows this is S3 (U = 0.80), not S4
(U = 0.81).  Dots mark medians, thick bars interquartile ranges and capped thin
bars full ranges.  Panel (b) uses E_H(S0) - E_H(strong), while panel (c) uses
D24(strong) - D24(S0); higher is better on both axes.

Typography and colour come from figstyle. The intended placement is 0.98 of
the manuscript text width; the font-size checks use that placement. Colour
encodes the weather case only (fs.WEATHER).

All headline numbers are asserted against the released data. Writes
fig_attainment_curves.(pdf|png). Deterministic.
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as fs

fs.apply(base_pt=9.1)

_HERE = os.path.dirname(os.path.abspath(__file__))

FRAC = 0.98          # intended standalone width relative to manuscript text


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

EX = _data_dir("S1_EXHAUSTIVE", "exhaustive_analysis")
FIGDIR = _figdir()

# horizon, legend label, figstyle weather key
CASES = [("2020", "Present", "present"),
         ("2050", "Mid-century", "mid"),
         ("2100", "Late-century", "late")]
COLS = ["windows_U_Factor", "groundfloor_thermal_resistance",
        "ext_walls_thermal_resistance", "roof_thermal_resistance"]
S0 = {"windows_U_Factor": 2.9, "groundfloor_thermal_resistance": 0.41,
      "ext_walls_thermal_resistance": 0.45, "roof_thermal_resistance": 0.48}
# component tick labels, in the COLS order; "Ground floor" is split over two
# lines so it does not eat the width of panel (b) at the printed size
COMP_LABEL = ["Windows", "Ground\nfloor", "Façade", "Roof"]
# envelope end values must equal Table 2's dD24 column; cliff pairs asserted
EXPECTED_END = {"2020": 1, "2050": -39, "2100": -57}
EXPECTED_CLIFF = {"2050": (14.702, 14.708, 7, -32), "2100": (10.714, 10.724, 6, -42)}
# (incumbent id, first post-cliff attainer id, its reduction, dd at the next step)
EXPECTED_DETAIL = {"2050": (525, 298, 14.785, -35), "2100": (519, 168, 10.943, -47)}

with open(EX + "all_configurations.csv") as fh:
    rows = list(csv.DictReader(fh))

# Keep one vertical scale so equal day-count differences have equal lengths.
fig = plt.figure(figsize=fs.size(FRAC, aspect=0.76), layout="constrained")
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])
axa = fig.add_subplot(gs[0, :])
axb = fig.add_subplot(gs[1, 0])
axc = fig.add_subplot(gs[1, 1], sharey=axb)
fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.06)

effects = {}   # (case, comp) -> (dE array, dD array)
gate = {}      # case -> (largest floor-free reduction, best there, best beyond)
for hz, lab, wkey in CASES:
    color = fs.WEATHER[wkey]
    mk = fs.WEATHER_MARKER[wkey]
    sub = [r for r in rows if r["horizon"] == hz]
    assert len(sub) == 625
    eh = np.array([float(r["annual_heating_energy_gj"]) for r in sub])
    dcount = np.array([float(r["days_below_24_c"]) for r in sub])
    ids = np.array([int(r["simulation_id"]) for r in sub])
    gfl = np.array([float(r["groundfloor_thermal_resistance"]) for r in sub])
    ib = next(i for i, r in enumerate(sub) if r["simulation_id"] == "1")
    red = eh[ib] - eh
    dd = dcount - dcount[ib]
    floor_free_mask = gfl == S0["groundfloor_thermal_resistance"]
    if hz in {"2050", "2100"}:
        assert dd[floor_free_mask].min() > dd[~floor_free_mask].max(), (
            hz, dd[floor_free_mask].min(), dd[~floor_free_mask].max())

    # ---- (a) exact attainment envelope (suffix max over reduction-sorted dd)
    order = np.argsort(red)
    rs, ds = red[order], dd[order]
    env = np.maximum.accumulate(ds[::-1])[::-1]
    assert int(env[-1]) == EXPECTED_END[hz], (hz, env[-1])
    if hz in EXPECTED_CLIFF:
        r1, r2, g1, g2 = EXPECTED_CLIFF[hz]
        i = int(np.searchsorted(rs, (r1 + r2) / 2))
        assert abs(rs[i - 1] - r1) < 5e-3 and abs(rs[i] - r2) < 5e-3, (hz, rs[i - 1], rs[i])
        assert int(env[i - 1]) == g1 and int(env[i]) == g2, (hz, env[i - 1], env[i])
        # the annotation's attainment facts: the incumbent is floor-free, the
        # post-cliff best is attained well beyond r2, and r2's own package is worse
        inc, att, att_red, dd_r2 = EXPECTED_DETAIL[hz]
        ji = int(np.where(ids == inc)[0][0])
        assert abs(red[ji] - r1) < 5e-3 and int(dd[ji]) == g1, (hz, red[ji], dd[ji])
        assert gfl[ji] == S0["groundfloor_thermal_resistance"], (hz, "incumbent has a floor state")
        beyond = red > red[ji]
        assert int(dd[beyond].max()) == g2, (hz, dd[beyond].max())
        sel = beyond & (dd == dd[beyond].max())
        ja = int(np.where(sel)[0][np.argmin(red[sel])])
        assert ids[ja] == att and abs(red[ja] - att_red) < 5e-4, (hz, ids[ja], red[ja])
        jr2 = int(np.where(beyond)[0][np.argmin(red[beyond])])
        assert abs(red[jr2] - r2) < 5e-3 and int(dd[jr2]) == dd_r2, (hz, red[jr2], dd[jr2])
    # Split the envelope by ground-floor family.  The unrestricted envelope is
    # the upper of these two lines.  A floor-free package remains admissible at
    # its exact reduction and disappears immediately above it, hence steps-pre.
    for floor_free, line_style in ((True, "-"), (False, ":")):
        family = floor_free_mask if floor_free else ~floor_free_mask
        fo = np.argsort(red[family])
        fr = red[family][fo]
        fd = dd[family][fo]
        fenv = np.maximum.accumulate(fd[::-1])[::-1]
        # g_F(r) is defined from r=0 even when the least-reduction member of F
        # already saves heating.  Extend the first plateau to the origin.
        if fr[0] > 0:
            fr = np.insert(fr, 0, 0.0)
            fenv = np.insert(fenv, 0, fenv[0])
        axa.plot(fr, fenv, color=color, lw=1.45 if floor_free else 1.15,
                 ls=line_style, drawstyle="steps-pre",
                 alpha=1.0 if floor_free else 0.8,
                 zorder=4 if floor_free else 3)
        # The terminal marker is the largest heating reduction in this family.
        axa.scatter([fr[-1]], [fenv[-1]], marker=mk, s=24,
                    facecolor="white" if floor_free else color,
                    edgecolor=color, lw=0.8, zorder=5)
        if floor_free:
            # the price of the gate: the floor-free optimum at its own limit,
            # against the best any insulated-floor package attains from there on
            beyond = (~floor_free_mask) & (red > fr[-1])
            gate[hz] = (float(fr[-1]), int(fenv[-1]), int(dd[beyond].max()))

    # ---- (b)/(c) matched component effects
    V = {c: np.array([float(r[c]) for r in sub]) for c in COLS}
    # thermally strongest state per component: min U for windows (S3, U = 0.80,
    # below the highest-index S4 at 0.81), max R for the three resistances
    strongest = {c: (min if c == "windows_U_Factor" else max)(set(V[c])) for c in COLS}
    for c in COLS:
        others = [o for o in COLS if o != c]
        m0 = {tuple(V[o][i] for o in others): i for i in range(625) if V[c][i] == S0[c]}
        m4 = {tuple(V[o][i] for o in others): i for i in range(625) if V[c][i] == strongest[c]}
        keys = sorted(set(m0) & set(m4))
        assert len(keys) == 125, (hz, c, len(keys))
        dE = np.array([eh[m0[k]] - eh[m4[k]] for k in keys])
        dD = np.array([dcount[m4[k]] - dcount[m0[k]] for k in keys])
        effects[(hz, c)] = (dE, dD)

# ground-floor medians the text quotes, asserted
assert round(float(np.median(effects[("2050", COLS[1])][0])), 2) == 2.39
assert round(float(np.median(effects[("2050", COLS[1])][1])), 0) == -40
assert round(float(np.median(effects[("2100", COLS[1])][0])), 2) == 2.29
assert round(float(np.median(effects[("2100", COLS[1])][1])), 0) == -53
for c in (COLS[0], COLS[2]):
    assert float(np.median(effects[("2100", c)][1])) == -12

# Under present weather D24 lies against its 365-day ceiling: 84 of the 125
# packages leaving the ground floor at state 0 are pinned there, and those are
# exactly the state-0 arm of the present ground-floor contrast.  The bounded
# threshold count has limited resolution in this weather case.
_present = [r for r in rows if r["horizon"] == "2020"]
_ceiling = sum(1 for r in _present
               if float(r["days_below_24_c"]) == 365.0
               and float(r["groundfloor_thermal_resistance"])
               == S0["groundfloor_thermal_resistance"])
assert _ceiling == 84, _ceiling
assert max(float(r["days_below_24_c"]) for r in _present) == 365.0

# The gate and its price, asserted.  The limit is the largest heating reduction
# a floor-free package delivers; the price is what the best attainable outcome
# loses on crossing it, and it is the quantity the panel exists to show.  It
# grows from two days today to 39 and 48 under the two warmer files.
GATE_LIMIT = {"2020": 17.3, "2050": 14.7, "2100": 10.7}
GATE_PRICE = {"2020": 2, "2050": 39, "2100": 48}
# Place the callouts below the upper curves, with a leader to each family limit.
# (x of the text, y of the text, horizontal alignment)
ANN = {"2100": (6.4, -3.0, "center"),
       "2020": (21.0, -3.0, "right"),
       "2050": (13.1, -3.0, "center")}
axa.axhline(0, color=fs.FAINT, lw=0.8, zorder=1)
for hz, lab, wkey in CASES:
    limit, free_best, insulated_best = gate[hz]
    price = free_best - insulated_best
    assert round(limit, 1) == GATE_LIMIT[hz], (hz, limit)
    assert price == GATE_PRICE[hz], (hz, free_best, insulated_best)
    tx, ty, ha = ANN[hz]
    # "drops by", not "costs": the manuscript carries a literal cost objective
    # I_C, and a cost-worded callout on a warm-side axis invites the wrong one.
    # A plain leader, not an arrowhead: the arrow identifies which limit the
    # label belongs to, and a head reads as though it drew the change itself,
    # which points the opposite way to the drop being named.
    axa.annotate(f"limit {limit:.1f} GJ/year\n"
                 f"$\\Delta D_{{24}}$ drops by {price} d",
                 xy=(limit, free_best), xytext=(tx, ty),
                 ha=ha, va="top", color=fs.WEATHER[wkey],
                 fontsize=10.1, linespacing=1.15,
                 arrowprops=dict(arrowstyle="-", lw=0.7,
                                 color=fs.WEATHER[wkey],
                                 shrinkA=2.0, shrinkB=2.5))
axa.set_ylim(-62, 12)
axa.set_yticks([-60, -40, -20, 0, 10])
axa.set_xlim(0, 21.2)
axa.grid(axis="y", color=fs.GRID, lw=0.5, zorder=0)
axa.grid(axis="x", visible=False)
axa.set_xlabel("Required heating reduction $r$ (GJ/year)")
axa.set_ylabel("Best attainable $\\Delta D_{24}$\n(days; higher is better)",
               linespacing=1.2)
axa.set_title("(a) Best attainable day-count change")

family_handles = [
    Line2D([], [], color=fs.INK, ls="-", lw=1.45, marker="o", ms=3.4,
           markerfacecolor="white", markeredgecolor=fs.INK,
           label="Ground floor at state 0"),
    Line2D([], [], color=fs.INK, ls=":", lw=1.15, marker="o", ms=3.4,
           markerfacecolor=fs.INK, markeredgecolor=fs.INK,
           label="Any insulated ground-floor state"),
]
axa.legend(handles=family_handles, loc="lower left", ncol=1, frameon=False,
           handlelength=2.4, borderaxespad=0.4)

# "thermally strongest state" is the manuscript's defined term and the title
# keeps it verbatim; at the printed width of a third-of-a-row panel it needs
# three lines, or (b)'s and (c)'s second lines run into each other and push the
# emitted canvas wider than the space LaTeX gives the figure.
for ax, col_idx, title, xlab in (
        (axb, 0, "(b) Heating reduction\nstrongest state vs state 0",
         r"$E_H(\mathrm{S0})-E_H(\mathrm{strong})$" "\n(GJ/year)"),
        (axc, 1, "(c) $D_{24}$ change\nstrongest state vs state 0",
         r"$D_{24}(\mathrm{strong})-D_{24}(\mathrm{S0})$" "\n"
         "(days; higher is better)")):
    y0 = np.arange(len(COLS))[::-1]
    for k, (hz, lab, wkey) in enumerate(CASES):
        color = fs.WEATHER[wkey]
        mk = fs.WEATHER_MARKER[wkey]
        # CASES follows fs.WEATHER_ORDER, and y increases upward, so the offset
        # must DECREASE with k for the rows inside a component group to read
        # Present -> Mid -> Late top to bottom, i.e. the same order as the
        # legend above.  Reversing this was what made the two disagree.
        off = (1 - k) * 0.24
        for ci, c in enumerate(COLS):
            v = effects[(hz, c)][col_idx]
            yy = y0[ci] + off
            ax.hlines(yy, v.min(), v.max(), color=color, lw=0.9, alpha=0.78, zorder=2)
            ax.vlines([v.min(), v.max()], yy - 0.035, yy + 0.035,
                      color=color, lw=0.8, alpha=0.78, zorder=2)
            ax.hlines(yy, np.percentile(v, 25), np.percentile(v, 75), color=color,
                      lw=2.2, zorder=3)
            ax.scatter([np.median(v)], [yy], s=16, color=color, edgecolor="white",
                       lw=0.6, zorder=4, marker=mk)
    ax.axvline(0, color=fs.FAINT, lw=0.8, zorder=1)
    for sep in (0.5, 1.5, 2.5):
        ax.axhline(sep, color=fs.GRID, lw=0.5, zorder=0)
    ax.set_ylim(-0.6, len(COLS) - 0.4)
    ax.set_yticks(y0)
    if ax is axb:
        ax.set_yticklabels(COMP_LABEL)
        ax.tick_params(axis="y", length=0)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlabel(xlab)
    ax.set_title(title)
    ax.grid(axis="x", color=fs.GRID, lw=0.5, zorder=0)
    ax.grid(axis="y", visible=False)

axb.set_xlim(0, 7.2)
axb.set_xticks([0, 2, 4, 6])
axc.set_xlim(-68, 8)
axc.set_xticks([-60, -40, -20, 0])

# one shared legend for all three panels -- (b) and (c) previously carried none
handles = [Line2D([], [], color=fs.WEATHER[w], ls="-", lw=1.3,
                  marker=fs.WEATHER_MARKER[w], ms=3.4, label=lab)
           for _, lab, w in CASES]
fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False,
           handlelength=2.4, columnspacing=2.4, borderaxespad=0.2)

fs.finalize(fig, FIGDIR + "fig_attainment_curves", frac=FRAC)
print("fig_attainment_curves written (continuous day scale; family limits, "
      "post-limit drops, ceiling census and matched medians asserted)")
