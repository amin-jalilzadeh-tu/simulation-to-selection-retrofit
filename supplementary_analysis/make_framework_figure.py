"""Generate the manuscript's framework figure (fig_workflow).

Output (to FIGDIR):
  fig_workflow.(pdf|png)  - the study framework: one simulated design space, two
                            ways of reaching a package, compared at three levels

Design rules this figure is held to, and which its predecessor broke:

1. NO RESULTS.  A framework figure states what was done, not what was found.
   Every number here is one the study CHOSE before running anything -- the size
   of the design space, the weather cases, the number of formulations and task
   designs, the preference profiles.  Nothing counted from an output appears
   (no Pareto-set sizes, no accuracy, no agreement rate), and no box carries an
   evaluative word.
2. NO SECTION POINTERS.  The predecessor tagged each box with a section number,
   which made it an illustrated table of contents.
3. NO STAGE NUMBERING AND NO TOOLCHAIN BRANDING.  Both are the conventions of
   the pipeline-provenance diagrams in this literature; neither carries meaning
   here.
4. PARALLEL, NOT DIVERGING, RAILS.  The two paths are drawn the same distance
   apart at every level.  Splaying them would encode the finding, which is
   rule 1 broken in geometry instead of in words.

Layout rules, which exist because breaking them produced visible defects:

5. A LEVEL'S TWO BOXES ARE THE SAME HEIGHT.  Sizing each box to its own line
   count left the three-line selection box towering over the one-line box
   opposite it, which reads as breakage rather than as a pair.
6. THE TWO KINDS OF ANNOTATION NEVER SHARE A REGION.  What happens ALONG a rail
   is centred on that rail and breaks its connector; what is compared ACROSS the
   rails lives in the channel between them.  The rail label used to be set
   beside its arrow, which pushed it into the channel and collided with the
   comparison label sitting in the same gap.

Deterministic; reads no data.  Authored at the width it occupies on the page,
so a nominal point size is the printed point size, and fs.finalize refuses to
write the figure if any lettering falls below the 7 pt artwork floor.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import figstyle as fs

fs.apply()


def _figdir():
    d = os.environ.get("FIGDIR", os.path.join(os.getcwd(), "figures_out"))
    os.makedirs(d, exist_ok=True)
    return d if d.endswith(os.sep) else d + os.sep


FIGDIR = _figdir()
FRAC = 0.98          # fraction of \linewidth the figure occupies

# --------------------------------------------------------------------------
# Structural identity.  These four hues are the namespace figstyle reserves for
# schematics: nothing here borrows a weather, objective, model or package hue.
# --------------------------------------------------------------------------
H_IN = fs.SCHEMATIC["input"]
H_EX = fs.SCHEMATIC["exact"]
H_SU = fs.SCHEMATIC["surrogate"]
H_CM = fs.SCHEMATIC["comparison"]
F_IN = fs.tint(H_IN, 0.86)
F_EX = fs.tint(H_EX, 0.86)
F_SU = fs.tint(H_SU, 0.86)

# --------------------------------------------------------------------------
# Horizontal grid, in layout units.  The canvas is 10 units wide and mapped
# isotropically onto the printed width, so one unit is the same length in x and
# y and box geometry can be reasoned about in inches.
#
# The three columns are NOT equal.  The left rail carries short noun phrases,
# the channel carries metric names, and the right rail carries the validation
# protocol; each width follows its own longest line rather than a symmetry that
# would break one of the three.
# --------------------------------------------------------------------------
GUT_X = 0.78                     # right edge of the level-name gutter
LX, LW = 0.90, 2.70              # left rail
CX, CW = 3.60, 2.85              # comparison channel
RX, RW = 6.45, 3.50              # right rail
KEY_X, VAL_X = 2.55, 2.73        # key/value split in the header and footer
LC = LX + LW / 2.0
RC = RX + RW / 2.0
CC = CX + CW / 2.0
HX, HW = 0.90, 9.05              # header block

PAD = 0.06                       # FancyBboxPatch pad, in units
INSET = 0.09                     # clear space demanded inside a box, in units

# --------------------------------------------------------------------------
# Type sizes.  Everything clears MIN_PT = 7.0 with margin: fs.finalize measures
# the PRINTED size, and a figure whose emitted bbox is narrower than the placed
# width would be scaled up, never down, by LaTeX.
# --------------------------------------------------------------------------
PT_HEAD_K = 7.6      # header row key
PT_HEAD_V = 7.8      # header row value
PT_BOX = 7.8         # left rail box text
PT_BOX_R = 7.6       # right rail, whose longest line is the binding one
PT_EDGE = 7.2        # what happens along a rail, between two of its boxes
PT_METRIC = 7.4      # what is compared across the channel
PT_LEVEL = 7.4       # level name in the gutter
PT_MATH = 8.7        # keeps rendered sub/superscripts at or above 6 pt

LINE_H = 0.215       # height of one line of body type, in units
_fitted = []         # (artist, (x, y, w, h)) regions policed by check_fits


# --------------------------------------------------------------------------
def _lines(text):
    return text.count("\n") + 1


def _pt(text, ordinary):
    """Raise only labels containing mathtext to the rendered-glyph floor."""
    return max(ordinary, PT_MATH) if "$" in text else ordinary


def box_height(*texts):
    """The height a rail box needs, sized on the longer of a level's pair."""
    n = max(_lines(t) for t in texts)
    return max(0.30 + LINE_H * n, 0.54)


def box(x, y, w, h, text, fc, ec, pt=PT_BOX):
    """A rail box, bottom-left anchored at (x, y)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={PAD}",
                                fc=fc, ec=ec, lw=0.9, zorder=2))
    t = ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center",
                fontsize=_pt(text, pt), color=fs.INK, linespacing=1.34, zorder=3)
    _fitted.append((t, (x, y, w, h)))


def rail(x, y_from, y_to, label=None, x0=None, x1=None):
    """A step along one rail.

    With a label, the connector is BROKEN and the label set in the break, on the
    rail's own axis: that keeps it inside its column, where it cannot reach the
    channel.  (x0, x1) bound the column, so check_fits can police the label
    against the space it is actually allowed.
    """
    if label is None:
        ax.add_patch(FancyArrowPatch((x, y_from), (x, y_to), arrowstyle="-|>",
                                     mutation_scale=8, lw=1.0, color=fs.MUTED,
                                     shrinkA=0, shrinkB=0, zorder=1))
        return
    mid = (y_from + y_to) / 2.0
    half = LINE_H * _lines(label) / 2.0 + 0.04
    ax.plot([x, x], [y_from, mid + half], color=fs.MUTED, lw=1.0, zorder=1)
    ax.add_patch(FancyArrowPatch((x, mid - half), (x, y_to), arrowstyle="-|>",
                                 mutation_scale=8, lw=1.0, color=fs.MUTED,
                                 shrinkA=0, shrinkB=0, zorder=1))
    t = ax.text(x, mid, label, ha="center", va="center", fontsize=PT_EDGE,
                color=fs.MUTED, style="italic", linespacing=1.3, zorder=3)
    _fitted.append((t, (x0, mid - half, x1 - x0, 2 * half)))


def spec_block(x, y, w, h, rows, hue, fill):
    """A key/value block: the inputs at the top, the outputs at the bottom.

    Both key and value are policed against their own column.  The predecessor
    set the keys flush to a hand-chosen x and the longest of them ("reference
    dwelling") overhung the box's left border -- the one collision visible in
    the printed figure.
    """
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={PAD}",
                                fc=fill, ec=hue, lw=0.9, zorder=2))
    row = (h - 2 * INSET) / len(rows)
    for i, (k, v) in enumerate(rows):
        ry = y + h - INSET - (i + 0.5) * row
        tk = ax.text(KEY_X, ry, k, ha="right", va="center", fontsize=PT_HEAD_K,
                     color=hue, zorder=3)
        tv = ax.text(VAL_X, ry, v, ha="left", va="center",
                     fontsize=_pt(v, PT_HEAD_V),
                     color=fs.INK, zorder=3)
        _fitted.append((tk, (x, ry - row / 2.0, KEY_X + INSET - x, row)))
        _fitted.append((tv, (VAL_X - INSET, ry - row / 2.0,
                             x + w - VAL_X + INSET, row)))


def check_fits():
    """Fail loudly if any label overruns the region it was allotted."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bad = []
    for t, (x, y, w, h) in _fitted:
        tb = t.get_window_extent(renderer=rend)
        x0, y0 = ax.transData.transform((x + INSET, y))
        x1, y1 = ax.transData.transform((x + w - INSET, y + h))
        over = {"left": (x0 - tb.x0) / fig.dpi, "right": (tb.x1 - x1) / fig.dpi,
                "bottom": (y0 - tb.y0) / fig.dpi, "top": (tb.y1 - y1) / fig.dpi}
        hit = {k: v for k, v in over.items() if v > 0}
        if hit:
            where = ", ".join(f"{k} by {v:.3f} in" for k, v in hit.items())
            bad.append(f"    {t.get_text().splitlines()[0]!r}: over {where}")
    if bad:
        raise ValueError("fig_workflow labels do not fit:\n" + "\n".join(bad))


# --------------------------------------------------------------------------
# Content.  Read this block, not the geometry, to see what the figure claims.
# --------------------------------------------------------------------------
HEADER = [
    ("dwelling", "1946–64 Dutch mid-terrace, three zones"),
    ("design space",
     "façade · roof · floor · windows, five states each → $5^4$ packages"),
    ("weather",
     "present TMYx · SSP2-4.5 mid-century · SSP5-8.5 late-century"),
    ("simulation", "EnergyPlus, every package under every weather case"),
    # The order is figstyle.OBJECTIVE_ORDER, which Eq. (5), every table and every
    # other figure use; and the warm-side entry is -D_24, because higher D_24 is
    # better and the all-minimisation vector carries its negation.  Printing
    # "D_24 ... all minimised" asserted the opposite direction to Fig. 2.
    ("objectives", "$E_H$ · $I_C$ · $I_G$ · $-D_{24}$, all minimised"),
    ("evaluation", "625-package composite (five folds) or 125-package held-out pool (one fold)"),
]

SRC_L = "simulated outputs\nfor the complete grid"
SRC_R = ("surrogate models\nfive formulations, two task designs\n"
         "package-grouped cross-validation\nover folds and seeds")

# What the framework yields, stated the same way the inputs are: the kinds of
# quantity produced and the grid they are produced over.  No value appears here
# either -- this is what the comparison reports, not what it reported.
FOOTER = [
    ("outputs", "prediction accuracy, Pareto-set recovery, selection agreement, regret"),
    ("reported for", "every formulation, task design and weather case"),
]

# (level name, left box, right box, channel metrics, left rail step, right rail step)
LEVELS = [
    ("objective\nvalues",
     "$E_H$, $D_{24}$ simulated\n$I_C$, $I_G$ calculated",
     "$E_H$, $D_{24}$ predicted\n$I_C$, $I_G$ calculated (exact-index)\nor predicted (all-predicted)",
     "MAE · RMSE · $R^2$\npooled and by case",
     None, None),
    ("Pareto\nsets",
     "reference Pareto set\nsame candidate set",
     "surrogate-derived Pareto set\nfrom the composite or held-out pool",
     "precision · recall · F1",
     "pairwise dominance",
     "range clipping,\nthen dominance"),
    ("package\nselection",
     "selected package $x^{*}$\nfive preference profiles\nDirichlet weight sweep",
     "selected package $\\hat{x}$",
     "selection agreement\nregret $R_w$\nreference-set membership",
     "normalise on own front,\nweighted sum",
     "normalise on own front,\nsame weights"),
]

# --------------------------------------------------------------------------
# Vertical layout, top-down.  Heights are in the same units as the widths.
# The gap between levels is set by what has to live in it: a two-line rail label
# in a broken connector, clear of the boxes above and below.
# --------------------------------------------------------------------------
M_TOP = M_BOT = 0.08
H_HEADER = 1.73
G_FORK = 0.34
H_SRC = box_height(SRC_L, SRC_R)
G_SRC = 0.38
G_LEVEL = 0.84
G_OUT = 0.42
H_FOOTER = 0.78
BAND = [box_height(lv[1], lv[2]) for lv in LEVELS]

Y_TOTAL = (M_TOP + H_HEADER + G_FORK + H_SRC + G_SRC
           + sum(BAND) + G_LEVEL * (len(LEVELS) - 1)
           + G_OUT + H_FOOTER + M_BOT)

PAD_IN = 0.05
W_IN = fs.LINEWIDTH_IN * FRAC
UNIT_IN = (W_IN - 2 * PAD_IN) / 10.0
ASPECT = (Y_TOTAL * UNIT_IN + 2 * PAD_IN) / W_IN

fig, ax = plt.subplots(figsize=fs.size(FRAC, aspect=ASPECT))
_fx = PAD_IN / W_IN
_fy = PAD_IN / (ASPECT * W_IN)
fig.subplots_adjust(left=_fx, right=1 - _fx, bottom=_fy, top=1 - _fy)
ax.set_xlim(0, 10.09)
ax.set_ylim(0, Y_TOTAL)
ax.axis("off")
ax.set_facecolor("none")

cur = Y_TOTAL - M_TOP            # a top-down cursor, in figure units

# ------------------------------------------------------------------ header
hy = cur - H_HEADER
spec_block(HX, hy, HW, H_HEADER, HEADER, H_IN, F_IN)
cur = hy

# -------------------------------------------------------------------- fork
sy = cur - G_FORK - H_SRC
stem = cur - G_FORK * 0.50
ax.plot([CC, CC], [cur, stem], color=fs.MUTED, lw=1.0, zorder=1)
ax.plot([LC, RC], [stem, stem], color=fs.MUTED, lw=1.0, zorder=1)
for xc in (LC, RC):
    ax.add_patch(FancyArrowPatch((xc, stem), (xc, sy + H_SRC), arrowstyle="-|>",
                                 mutation_scale=8, lw=1.0, color=fs.MUTED,
                                 shrinkA=0, shrinkB=0, zorder=1))
box(LX, sy, LW, H_SRC, SRC_L, F_EX, H_EX)
box(RX, sy, RW, H_SRC, SRC_R, F_SU, H_SU, pt=PT_BOX_R)
cur = sy

# ------------------------------------------------------------------ levels
for i, (level, left, right, metric, el, er) in enumerate(LEVELS):
    gap = G_SRC if i == 0 else G_LEVEL
    top = cur - gap
    h = BAND[i]
    bot = top - h
    mid = top - h / 2.0

    rail(LC, cur, top, el, x0=LX, x1=LX + LW)
    rail(RC, cur, top, er, x0=RX, x1=RX + RW)

    box(LX, bot, LW, h, left, F_EX, H_EX)
    box(RX, bot, RW, h, right, F_SU, H_SU, pt=PT_BOX_R)

    ax.text(GUT_X, mid, level, ha="right", va="center", fontsize=PT_LEVEL,
            color=H_CM, linespacing=1.3, zorder=3)

    # The comparison arrow sits ON the axis the level's two boxes share, and its
    # label stacks upward from there.  The channel carries nothing else, so the
    # label can rise into the gap above without meeting the rail labels, which
    # are confined to their own columns.
    ax.add_patch(FancyArrowPatch((CX + 0.14, mid), (CX + CW - 0.14, mid),
                                 arrowstyle="<|-|>", mutation_scale=7, lw=1.0,
                                 color=H_CM, shrinkA=0, shrinkB=0, zorder=3))
    metric_pt = _pt(metric, PT_METRIC)
    th = LINE_H * _lines(metric) * metric_pt / PT_METRIC
    t = ax.text(CC, mid + 0.15 + th / 2.0, metric, ha="center", va="center",
                fontsize=metric_pt, color=H_CM,
                linespacing=1.32, zorder=3)
    _fitted.append((t, (CX, mid + 0.15, CW, th)))

    last_mid = mid
    cur = bot

# ----------------------------------------------------------------- outputs
fy = cur - G_OUT - H_FOOTER
ax.add_patch(FancyArrowPatch((CC, last_mid), (CC, fy + H_FOOTER),
                             arrowstyle="-|>", mutation_scale=8, lw=1.0,
                             color=H_CM, shrinkA=0, shrinkB=0, zorder=1))
spec_block(HX, fy, HW, H_FOOTER, FOOTER, H_CM, fs.tint(H_CM, 0.90))

check_fits()
fs.finalize(fig, FIGDIR + "fig_workflow", frac=FRAC)
print("fig_workflow written")
