"""Graphical-abstract DRAFT (author mock-up; NOT for direct submission).

The Energy Reports guide for authors prohibits generative-AI or AI-assisted
tools in graphical-abstract production, and an AI assistant helped lay this
draft out. It therefore carries a visible draft footer and serves only as a
content/layout specification for the authors to recreate (or to decide to omit
the optional graphical abstract altogether).

Deterministic matplotlib vector artwork from author-verified manuscript numbers.
Canvas is nominally 13 x 5 cm; with tight cropping the PDF prints at roughly
12.7 x 5.4 cm and the 300-dpi PNG is about 1501 x 638 px, above the journal's
1328 x 531 px minimum. Writes graphical_abstract.(pdf|png) to FIGDIR.

Two parallel evidence paths, matching the manuscript's structure: the exact
enumeration carries the heating/warm-side trade-off finding on its own, and the
surrogate comparison against that exact reference carries the decision-fidelity
finding (pooled R^2 >= 0.988 yet at most 20 of 45 profile selections reproduced).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _figdir():
    d = os.environ.get("FIGDIR", os.path.join(os.getcwd(), "figures_out"))
    os.makedirs(d, exist_ok=True)
    return d if d.endswith(os.sep) else d + os.sep

FIGDIR = _figdir()

TEAL, ORANGE, INK = "#2a9d8f", "#b3541e", "#1c2b36"

fig, ax = plt.subplots(figsize=(5.118, 1.969))  # 13 x 5 cm
ax.set_xlim(0, 13)
ax.set_ylim(0, 5)
ax.axis("off")


def box(x, y, w, h, text, fc, ec="0.35", fs=6.2, tc=INK, weight="normal", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight=weight, linespacing=1.25)


def arrow(x1, y1, x2, y2, color="0.3"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=8, lw=1.0, color=color))


# ---- the study
box(0.15, 1.35, 2.7, 2.3,
    "Dutch 1946–1964\nmid-terrace dwelling\n625 envelope packages\n× 3 weather cases\n= 1,875 simulations",
    "#dbe9f6", fs=5.8)

# ---- evidence path 1: exact enumeration -> trade-off finding
box(3.5, 2.75, 3.9, 1.5,
    "Exact enumeration\nall fronts and selections known", "#d9efe8", ec=TEAL, fs=5.9)
box(8.15, 2.75, 4.7, 1.5,
    "Late-century case: the minimum-heating\npackage records 57 fewer below-24 °C\ndays than the unretrofitted baseline",
    "white", ec=TEAL, fs=5.7, lw=1.3)
arrow(2.9, 3.15, 3.5, 3.5)
arrow(7.45, 3.5, 8.13, 3.5)

# ---- evidence path 2: surrogates vs that exact reference -> fidelity finding
box(3.5, 0.65, 3.9, 1.5,
    "Surrogates vs the exact reference:\nsame front, same package?", "#fdeadb",
    ec=ORANGE, fs=5.7)
box(8.15, 0.65, 4.7, 1.5,
    "pooled $R^2 \\geq 0.988$, yet at most\n20 of 45 profile selections\nreproduced",
    "white", ec=ORANGE, fs=5.7, lw=1.3)
arrow(2.9, 1.95, 3.5, 1.45)
arrow(7.45, 1.4, 8.13, 1.4)
arrow(5.45, 2.72, 5.45, 2.2, color=TEAL)  # the exact track is the reference

ax.text(6.5, 4.72, "Prediction accuracy does not guarantee the same retrofit recommendation",
        ha="center", va="center", fontsize=7.2, color=INK, fontweight="bold")
ax.text(6.5, 0.14, "draft layout for author recreation — not for submission",
        ha="center", va="center", fontsize=4.6, color="0.55", style="italic")

fig.tight_layout(pad=0.15)
for ext in ("pdf", "png"):
    fig.savefig(FIGDIR + f"graphical_abstract.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("graphical_abstract written (13 x 5 cm, 300 dpi, two evidence paths)")
