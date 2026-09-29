"""SMAA-style MCDA figures on the exact Pareto fronts (Supplementary Figures S1.1 and S1.2).

Outputs (to FIGDIR):
  fig_weight_regions.(pdf|png)     - which package wins where in weight space: two
                                     ternary slices of the four-weight simplex at
                                     fixed w_G, per weather case; regions >= ~3% of
                                     a panel carry their simulation id at the region
                                     medoid; distinct package hues for the largest
                                     winners, light grey otherwise; packages that
                                     are not uniquely selectable win no region
  fig_rank_acceptability.(pdf|png) - SMAA rank-acceptability heatmap b_i^r (ranks
                                     1-5, single neutral ramp) over 10,000 uniform
                                     weight draws, top packages per case; the
                                     remaining probability lies beyond rank 5
Deterministic (seed 42, same draws as mcda_supportedness_sweep.py).

Typography/palette: authored at final printed width through figstyle, so nominal
point sizes are printed point sizes; fills come from the package-identity
namespace only (never the reserved weather hues).
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch

import figstyle as fs

fs.apply()

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

EX = _data_dir("S1_EXHAUSTIVE", "exhaustive_analysis")
FIGDIR = _figdir()

HORIZONS = ["2020", "2050", "2100"]
HZLABEL = {"2020": "Present", "2050": "Mid-century", "2100": "Late-century"}

with open(EX + "pareto_fronts.csv") as _fh:
    rows = list(csv.DictReader(_fh))
DATA = {}
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
    DATA[hz] = (ids, Fn)

# Filled winner regions on a fine grid (no marker stippling), with white region
# boundaries.  The fill encodes PACKAGE IDENTITY, so every hue is drawn from the
# package-identity ramp: the do-nothing package keeps the fixed baseline grey,
# the semantically named selections take four fixed package hues, every other
# winner >= 1% of a panel cycles the remaining package hues, and smaller regions
# take the sub-threshold grey.  Direct in-region id labels are the key; the
# legend carries only the semantic entries.
SEMANTIC = {
    1:   fs.BASELINE_FILL,            # do-nothing package
    25:  fs.PACKAGE_CATEGORICAL[6],   # equal-weight selection, present
    13:  fs.PACKAGE_CATEGORICAL[2],   # equal-weight selection, warmer
    15:  fs.PACKAGE_CATEGORICAL[4],   # D24-priority selection, warmer
    550: fs.PACKAGE_CATEGORICAL[7],   # E_H-priority selections
    395: fs.PACKAGE_CATEGORICAL[7],
    250: fs.PACKAGE_CATEGORICAL[7],
}
OTHER_FILLS = [fs.PACKAGE_CATEGORICAL[i] for i in (0, 1, 3, 5, 8, 9)]
MINOR_GREY = fs.MINOR_FILL


def tern_xy(w):  # barycentric (wE, wC, wD) -> 2D
    wE, wC, wD = w
    s = wE + wC + wD
    wE, wC, wD = wE / s, wC / s, wD / s
    x = wC + 0.5 * wD
    y = (np.sqrt(3) / 2) * wD
    return x, y


def select_ws(hz, w4):
    """Named-profile selection (isclose rtol=0 tie rule) at an exact weight vector."""
    ids, Fn = DATA[hz]
    scores = Fn @ np.asarray(w4)
    best = np.min(scores)
    cand = np.where(np.isclose(scores, best, rtol=0.0, atol=1e-12))[0]
    return int(ids[cand[np.argmin(ids[cand])]])


# The map must agree with the manuscript's selections at every star point.
EXPECTED_STAR = {
    ("E", "2020"): 550, ("E", "2050"): 395, ("E", "2100"): 250,
    ("C", "2020"): 1, ("C", "2050"): 1, ("C", "2100"): 1,
    ("D", "2020"): 25, ("D", "2050"): 15, ("D", "2100"): 15,
    ("Bal", "2020"): 25, ("Bal", "2050"): 13, ("Bal", "2100"): 13,
}
STAR_W = {"E": (0.7, 0.1, 0.1, 0.1), "C": (0.1, 0.7, 0.1, 0.1),
          "D": (0.1, 0.1, 0.1, 0.7), "Bal": (0.25, 0.25, 0.25, 0.25)}
for (lab, hz), exp in EXPECTED_STAR.items():
    got = select_ws(hz, STAR_W[lab])
    assert got == exp, (lab, hz, got, exp)

SLICES = [0.10, 0.25]
NX, NY = 560, 490
H = np.sqrt(3) / 2
gx = np.linspace(0.0, 1.0, NX)
gy = np.linspace(0.0, H, NY)
GX, GY = np.meshgrid(gx, gy)
wD_n = GY / H
wC_n = GX - 0.5 * wD_n
wE_n = 1.0 - wC_n - wD_n
valid = (wE_n > -1e-9) & (wC_n > -1e-9) & (wD_n > -1e-9)

panel_fields, share = [], defaultdict(float)
for si, wg in enumerate(SLICES):
    rem = 1.0 - wg
    W4 = np.stack([rem * wE_n[valid], rem * wC_n[valid],
                   np.full(valid.sum(), wg), rem * wD_n[valid]], axis=1)
    for hi, hz in enumerate(HORIZONS):
        ids, Fn = DATA[hz]
        order = np.argsort(ids)
        winner = ids[order][np.argmin((W4 @ Fn.T)[:, order], axis=1)]
        ID = np.full(GX.shape, -1, dtype=int)
        ID[valid] = winner
        panel_fields.append((si, hi, ID))
        for sid, cnt in zip(*np.unique(winner, return_counts=True)):
            share[sid] += cnt

muted_ids = [s for s in sorted(share, key=lambda s: -share[s]) if s not in SEMANTIC]
COLOR = dict(SEMANTIC)
total = sum(share.values())
for i, sid in enumerate(muted_ids):
    COLOR[sid] = (OTHER_FILLS[i % len(OTHER_FILLS)]
                  if share[sid] / total >= 0.01 / 6 else MINOR_GREY)

FRAC_REGIONS = 1.0
LBL_PT = 7.0            # printed size of the in-region id labels
TRI_PAD = 0.012         # keep a label box off the triangle edge


def _half_extent(txt, ax, fig):
    """Half-width/half-height of a label, in ternary data units."""
    w_pt = 0.5 * 0.62 * LBL_PT * len(str(txt)) + 1.3   # digit advance + halo
    h_pt = 0.5 * LBL_PT + 1.3
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((0.0, 0.0))
    x1, y1 = inv.transform((w_pt * fig.dpi / 72.0, h_pt * fig.dpi / 72.0))
    return abs(x1 - x0), abs(y1 - y0)


def _clamp_into_triangle(x, y, hw, limit=0.06):
    """Nudge a label box back inside the triangle, but only a little."""
    lo, hi = 0.5 * y / H + hw + TRI_PAD, 1.0 - 0.5 * y / H - hw - TRI_PAD
    if lo >= hi:
        return x
    xc = min(max(x, lo), hi)
    return xc if abs(xc - x) <= limit else x


def _overlap(a, b):
    return (abs(a[0] - b[0]) < a[2] + b[2]) and (abs(a[1] - b[1]) < a[3] + b[3])


def _leader_hits(x0, y0, x1, y1, placed):
    """Does the callout's leader run through a label that is already placed?"""
    n = 0
    for t in np.linspace(0.12, 0.88, 9):
        px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        n += sum(1 for q in placed
                 if abs(px - q[0]) < q[2] and abs(py - q[1]) < q[3])
    return n


def _place(x0, y0, hw, hh, placed, offsets, clamp=True, inside=None,
           leader_over=None):
    """Cheapest offset: clear of other labels, on its own region, short leader."""
    best, best_cost = None, None
    for dx, dy in offsets:
        x, y = x0 + dx, y0 + dy
        if clamp:
            x = _clamp_into_triangle(x, y, hw)
        box = (x, y, hw, hh)
        cost = 10.0 * sum(1 for q in placed if _overlap(box, q))
        if inside is not None and not inside(x, y):
            cost += 4.0
        if leader_over is not None:
            cost += 3.0 * _leader_hits(x0, y0, x, y, leader_over)
        cost += 0.5 * (abs(dx) + abs(dy))
        if best_cost is None or cost < best_cost:
            best, best_cost = (x, y), cost
        if best_cost == 0.0:
            break
    return best


# Region labels are nudged only when a star or another label already occupies the
# medoid; the callouts for small semantically named regions pick the first
# direction that stays clear.  Both searches are deterministic.
REGION_OFFSETS = [(0.0, 0.0), (0.0, -0.075), (0.0, 0.075), (-0.075, -0.045),
                  (0.075, -0.045), (-0.075, 0.045), (0.075, 0.045), (0.0, -0.125)]
CALLOUT_OFFSETS = [(-0.17, 0.09), (-0.20, 0.0), (0.17, 0.09), (-0.12, 0.16),
                   (0.20, 0.0), (0.12, 0.16), (0.17, -0.09), (-0.17, -0.09)]

fig, axes = plt.subplots(len(SLICES), 3,
                         figsize=fs.size(FRAC_REGIONS, aspect=0.78))
fig.subplots_adjust(left=0.052, right=0.995, top=0.945, bottom=0.190,
                    wspace=0.055, hspace=0.03)
for si, hi, ID in panel_fields:
    ax = axes[si, hi]
    rgb = np.ones(ID.shape + (3,))
    for sid in np.unique(ID[ID >= 0]):
        rgb[ID == sid] = to_rgb(COLOR[sid])
    ax.imshow(rgb, origin="lower", extent=(0, 1, 0, H), zorder=1,
              interpolation="nearest")
    npanel = int((ID >= 0).sum())
    wg = SLICES[si]
    hz = HORIZONS[hi]
    ax.set_xlim(-0.055, 1.055)
    ax.set_ylim(-0.115, 0.955)
    ax.set_aspect("equal")
    slice_stars = ["E", "C", "D"] if abs(wg - 0.10) < 1e-9 else ["Bal"]
    # the RENDERED map must agree with Table 3 at every star: assert the winner
    # of the grid cell nearest each star point, not only the analytic selection
    for slab in slice_stars:
        wv = STAR_W[slab]
        sx, sy = tern_xy((wv[0], wv[1], wv[3]))
        cell = ID[np.argmin(np.abs(gy - sy)), np.argmin(np.abs(gx - sx))]
        assert cell == EXPECTED_STAR[(slab, hz)], (slab, hz, int(cell))

    regions = []
    for sid in np.unique(ID[ID >= 0]):
        m = ID == sid
        ax.contour(GX, GY, m.astype(float), levels=[0.5], colors="white",
                   linewidths=0.7, zorder=2)
        py, px = np.where(m)
        P = np.stack([GX[py, px], GY[py, px]], axis=1)
        med = P[np.argmin(((P - P.mean(0)) ** 2).sum(axis=1))]
        regions.append((int(sid), m.sum() / npanel, float(med[0]), float(med[1])))

    # the stars and their profile letters are placed first: they are fixed points
    placed, placed_text = [], []
    for slab in slice_stars:
        wv = STAR_W[slab]
        sx, sy = tern_xy((wv[0], wv[1], wv[3]))
        shw, shh = _half_extent(slab, ax, fig)
        mhw, mhh = _half_extent("0", ax, fig)
        placed.append((sx, sy, mhw, mhh))                       # marker
        off_x, off_y = _half_extent("00", ax, fig)
        placed.append((sx + off_x + shw, sy + off_y, shw, shh))  # letter
        placed_text.append(placed[-1])
        ax.plot(sx, sy, "*", ms=7, color=fs.INK, zorder=7)
        ax.annotate(slab, (sx, sy), xytext=(5, 3), textcoords="offset points",
                    fontsize=LBL_PT, zorder=7,
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])

    for sid, pshare, mx, my in sorted(regions, key=lambda t: -t[1]):
        hw, hh = _half_extent(sid, ax, fig)
        if pshare >= 0.03:  # label the region at (or near) its medoid
            def _on_region(x, y, _sid=sid):
                iy = int(np.argmin(np.abs(gy - y)))
                ix = int(np.argmin(np.abs(gx - x)))
                return bool(ID[iy, ix] == _sid)
            x, y = _place(mx, my, hw, hh, placed, REGION_OFFSETS,
                          inside=_on_region)
            ax.text(x, y, str(sid), fontsize=LBL_PT, ha="center", va="center",
                    color=fs.INK, zorder=6,
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
            placed.append((x, y, hw, hh))
            placed_text.append((x, y, hw, hh))
        elif sid in SEMANTIC and pshare > 0:  # callout for small semantic winners
            x, y = _place(mx, my, hw, hh, placed, CALLOUT_OFFSETS, clamp=False,
                          leader_over=placed_text)
            x = min(max(x, -0.05 + hw), 1.05 - hw)
            y = min(max(y, -0.10 + hh), 0.95 - hh)
            ax.annotate(str(sid), xy=(mx, my), xytext=(x, y),
                        fontsize=LBL_PT, color=fs.INK, ha="center", va="center",
                        zorder=7,
                        arrowprops=dict(arrowstyle="-", lw=0.6, color=fs.MUTED,
                                        shrinkA=2.0, shrinkB=1.0),
                        path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
            placed.append((x, y, hw, hh))
            placed_text.append((x, y, hw, hh))

    tri = plt.Polygon([(0, 0), (1, 0), (0.5, H)], fill=False, ec=fs.INK, lw=0.9,
                      zorder=3)
    ax.add_patch(tri)
    ax.text(0.0, -0.035, "$w_E$", fontsize=9.0, ha="center", va="top")
    ax.text(1.0, -0.035, "$w_C$", fontsize=9.0, ha="center", va="top")
    ax.text(0.5, H + 0.025, "$w_D$", fontsize=9.0, ha="center", va="bottom")
    if si == 0:
        ax.set_title(HZLABEL[hz], fontsize=9.0, pad=5)
    ax.axis("off")

# Row labels ride in the left margin, rotated, so the panels keep the full width.
# Nine-point weight labels keep their subscripts above the rendered-glyph floor.
for si, wg in enumerate(SLICES):
    pos = axes[si, 0].get_position()
    fig.text(0.006, 0.5 * (pos.y0 + pos.y1),
             f"slice $w_G$ = {wg:.2f}\n($w_E{{+}}w_C{{+}}w_D$ = {1 - wg:.2f})",
             rotation=90, ha="left", va="center", fontsize=9.0)

# Plain profile labels remain legible at the compact legend size.
legend = [Patch(color=SEMANTIC[1], label="1 (baseline)"),
          Patch(color=SEMANTIC[25], label="25 (equal-weight, present)"),
          Patch(color=SEMANTIC[13], label="13 (equal-weight, warmer)"),
          Patch(color=SEMANTIC[15], label="15 (D24-priority, warmer)"),
          Patch(color=SEMANTIC[550], label="Heating-priority (550/395/250)"),
          Patch(color=OTHER_FILLS[0], label="other winners (labelled if ≥3%)"),
          Patch(color=MINOR_GREY, label="minor winners")]
fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.005),
           ncol=3, fontsize=7.5, frameon=False,
           title="Winning package at each weight vector\n(region areas are"
                 " conditional on the displayed slice, not full-simplex"
                 " acceptabilities)",
           title_fontsize=7.5, handlelength=1.5, handleheight=0.9,
           columnspacing=1.1, handletextpad=0.5, labelspacing=0.4,
           borderaxespad=0.0)
fs.finalize(fig, FIGDIR + "fig_weight_regions", frac=FRAC_REGIONS)
print("fig_weight_regions written; star winners verified; semantic ids:",
      sorted(SEMANTIC), "| muted ids:", [int(s) for s in muted_ids[:10]])

# ------------------------------------------------------------ rank acceptability
# Heatmap of b_i^r for ranks 1-5; the probability beyond rank 5 is omitted and
# stated in the caption, so the uninformative remainder no longer dominates the
# graphic.  The ramp is neutral: acceptability is a magnitude, and every hue in
# this manuscript is spoken for by a categorical namespace.
ACC_CMAP = LinearSegmentedColormap.from_list(
    "acceptability", ["#FFFFFF", fs.FAINT, fs.MUTED, fs.INK])

rng = np.random.default_rng(42)
W = rng.dirichlet(np.ones(4), size=10000)
NR = 5  # ranks shown

panel = {}
vmax = 0.0
for hz in HORIZONS:
    ids, Fn = DATA[hz]
    scores = W @ Fn.T
    order = np.argsort(ids)  # deterministic tie-break by id
    sorted_scores = scores[:, order]
    ranks = np.empty_like(sorted_scores, dtype=int)
    rk = np.argsort(sorted_scores, axis=1, kind="stable")
    for d in range(rk.shape[0]):
        ranks[d, rk[d]] = np.arange(rk.shape[1])
    ids_o = ids[order]
    b = {}
    for ci, sid in enumerate(ids_o):
        cnt = np.bincount(np.minimum(ranks[:, ci], NR), minlength=NR + 1)
        b[sid] = cnt / 100.0  # percent
    top = sorted(b, key=lambda s: (-b[s][0], np.dot(b[s], np.arange(NR + 1))))[:8]
    M = np.array([[b[s][r] for r in range(NR)] for s in top])
    panel[hz] = (top, M)
    vmax = max(vmax, M.max())

EQ_SEL = {"2020": 25, "2050": 13, "2100": 13}  # the case's equal-weight selection
FRAC_RANK = 1.0
fig, axes = plt.subplots(1, 3, figsize=fs.size(FRAC_RANK, aspect=0.40))
fig.subplots_adjust(left=0.115, right=0.875, wspace=0.42, bottom=0.185, top=0.875)
for ax, hz in zip(axes, HORIZONS):
    top, M = panel[hz]
    im = ax.imshow(M, cmap=ACC_CMAP, vmin=0, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center",
                        va="center", fontsize=7.0,
                        color="white" if v > 0.60 * vmax else fs.INK)
    ax.set_xticks(range(NR))
    ax.set_xticklabels([f"{r + 1}" for r in range(NR)], fontsize=8.0)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{s} (base)" if s == 1 else
                        f"{s} (eq-wt)" if s == EQ_SEL[hz] else str(s)
                        for s in top], fontsize=7.0)
    ax.set_xlabel("Rank $r$", fontsize=8.5)
    ax.set_title(HZLABEL[hz], fontsize=9.0, pad=4)
    ax.tick_params(length=0, pad=1.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
axes[0].set_ylabel("Package (simulation id)", fontsize=8.5, labelpad=2)
cax = fig.add_axes([0.892, 0.185, 0.014, 0.69])  # reserved; no panel squeeze
cb = fig.colorbar(im, cax=cax)
cb.set_label("Rank acceptability\n(%)", fontsize=8.0, labelpad=4)
cb.ax.tick_params(labelsize=7.0)
cb.outline.set_visible(False)
fs.finalize(fig, FIGDIR + "fig_rank_acceptability", frac=FRAC_RANK)
print("fig_rank_acceptability written (heatmap, ranks 1-5)")
