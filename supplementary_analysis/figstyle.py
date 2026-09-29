"""Shared figure style for the envelope-retrofit manuscript.

Two jobs, both of which the twelve generators previously did inconsistently or
not at all:

1. TYPOGRAPHY AT FINAL SIZE.  Every generator authored its figure at an
   arbitrary width and let LaTeX downscale it, so nominal 8 pt labels landed at
   4.5-5.7 pt on the page.  `size()` authors the figure at the width it will
   actually occupy, so a nominal point size is the printed point size.
   `finalize()` checks TWICE and writes nothing unless both pass: ordinary
   lettering vs MIN_PT (7.0), and the emitted PDF's own font operators vs
   MIN_GLYPH_PT (6.0).  The second check exists because mathtext shrinks
   sub/superscripts below their artist's nominal size, which the first cannot
   see.  Pass standalone=True for artwork placed at a fixed physical size
   (the journal graphical abstract) -- otherwise the guard assumes the figure
   is stretched to frac*linewidth and will inflate its glyph sizes to match.

2. ONE MEANING PER COLOUR, ACROSS THE WHOLE SET.  Previously blue meant
   present weather (Fig 2), the equal-weight package (Fig 3), neural networks
   (Fig 4), acceptability magnitude (Fig 6) and the baseline package (S1.1).
   Colour is allocated here by namespace, and the namespaces do not overlap.

Palette rule
------------
The three "loud" hues -- blue, orange, red -- are RESERVED for the weather
cases and appear nowhere else, because the weather case is the one distinction
the reader must carry across figures.  Everything else is allocated away from
them:

    WEATHER        blue -> orange -> red        (reads as warming)
    OBJECTIVE      purple / teal / olive / pink
    MODEL_FAMILY   navy / green  + marker shape (redundant encoding)
    TASK_DESIGN    light -> dark neutral ramp   (it is a paired move, not two
                                                 categories; the arrow carries
                                                 the meaning)
    SELECTION      grey baseline + the OBJECTIVE hue each profile prioritises

All hues are checked for deuteranope/protanope separation within a namespace;
shape and line style reinforce colour wherever a namespace has >2 members.

Usage
-----
    import figstyle as fs
    fs.apply()
    fig, ax = plt.subplots(figsize=fs.size(0.98, aspect=0.34))
    ...
    fs.finalize(fig, FIGDIR + "fig_attainment_leverage", frac=0.98)
"""
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Page geometry.  Measured from the compiled manuscript (output/pdf/main.pdf):
# the cas-sc text block is 465 pt wide.  Not guessed from the class file.
# ---------------------------------------------------------------------------
LINEWIDTH_IN = 465.0 / 72.0     # = 6.458 in
MIN_PT = 7.0        # Elsevier artwork floor, applied to ordinary lettering
MIN_GLYPH_PT = 6.0  # floor for RENDERED glyphs, which mathtext shrinks below
                    # their artist's nominal size: matplotlib sets a subscript
                    # at 0.7x and a nested subscript at 0.49x, so a 9 pt
                    # $D_{T_u}$ prints its innermost glyph at 4.41 pt.  The
                    # artist-level check cannot see this -- only the emitted
                    # PDF can -- so both checks run.

# ---------------------------------------------------------------------------
# Namespaced palettes
# ---------------------------------------------------------------------------
WEATHER = {
    "present": "#3B75AF",
    "mid":     "#B85C00",
    "late":    "#8E1B2B",
}
WEATHER_ORDER = ("present", "mid", "late")
WEATHER_LABEL = {
    "present": "Present (TMYx)",
    "mid":     "Mid-century (SSP2-4.5, 2050)",
    "late":    "Late-century (SSP5-8.5, 2080)",
}
WEATHER_MARKER = {"present": "o", "mid": "s", "late": "^"}

OBJECTIVE = {
    "E_H":  "#6E4B9E",   # purple
    "I_C":  "#357F74",   # teal
    "I_G":  "#7F9E3C",   # olive
    "D_24": "#A85292",   # magenta (4.6:1 on white; text-safe)
}
# One fixed order everywhere -- matches Eq. (5) and every table.
OBJECTIVE_ORDER = ("E_H", "I_C", "I_G", "D_24")
OBJECTIVE_LABEL = {
    "E_H":  r"$E_H$",
    "I_C":  r"$I_C$",
    "I_G":  r"$I_G$",
    "D_24": r"$D_{24}$",
}

MODEL_FAMILY = {
    "network": "#2F5C8A",
    "tree":    "#2E8B72",
}
MODEL_MARKER = {"network": "o", "tree": "D"}

# A before/after of the same quantity, so value not hue.
TASK_DESIGN = {
    "all_predicted": "#9E9E9E",
    "exact_index":   "#333333",
}
TASK_FILL = {"all_predicted": "none", "exact_index": "full"}
TASK_MARKER = {"all_predicted": "o", "exact_index": "s"}

# Selection roles inherit the objective hue they prioritise, so a reader who
# has learned "purple = heating" reads the E_H-priority selection for free.
SELECTION = {
    "baseline":     "#808080",
    "equal_weight": OBJECTIVE["I_C"],
    "E_H_priority": OBJECTIVE["E_H"],
    "D_24_priority": OBJECTIVE["D_24"],
}
SELECTION_MARKER = {"baseline": "o", "equal_weight": "s",
                    "E_H_priority": "^", "D_24_priority": "D"}

# Component colours -- for the case/design-space figure and any component-keyed
# panel.  Deliberately outside the reserved weather hues.
COMPONENT = {
    "windows": "#4C9BB0",   # cyan-teal
    "floor":   "#8C6D3F",   # earth brown -- the ground floor sits on soil
    "facade":  "#9B7FB8",   # lilac
    "roof":    "#5B8C5A",   # moss green
}

# Qualitative ramp for package identity (e.g. the ternary winner regions), which
# needs ~10 mutually distinct fills.  Excludes the reserved weather hues so a
# package region is never mistaken for a weather case.
PACKAGE_CATEGORICAL = (
    "#4C9BB0", "#9B7FB8", "#5B8C5A", "#8C6D3F", "#C77CB0",
    "#7F9E3C", "#3A8A7D", "#6E4B9E", "#A0785A", "#6D8EA0",
)
BASELINE_FILL = "#B8B8B8"   # the do-nothing package, everywhere
MINOR_FILL = "#EDEDED"      # sub-threshold / unlabelled regions

# Paired sampling designs in the SMAA panel -- same quantity, two sampling
# regimes, so value not hue (as with TASK_DESIGN, and deliberately distinct
# from it so the two pairs never collide across figures).
SAMPLING_DESIGN = {"weights_only": "#8C8C8C", "weights_threshold": "#2B2B2B"}
SAMPLING_MARKER = {"weights_only": "o", "weights_threshold": "o"}
SAMPLING_FILL = {"weights_only": "none", "weights_threshold": "full"}

# Schematic diagrams (workflow, architecture) need structural identity that means
# none of weather/objective/model/task/selection.  Three agents reached for
# PACKAGE_CATEGORICAL instead, landing next to the reserved weather hues.
SCHEMATIC = {
    "input":      "#6D8EA0",   # slate  -- data entering the study
    "exact":      "#3F7D6B",   # pine   -- the enumerated reference track
    "surrogate":  "#A0785A",   # umber  -- the predicted track
    "comparison": "#7A6B93",   # heather-- where the two tracks are compared
}

# Sequential ramp for a magnitude (rank acceptability, any heatmap).  Built from
# the module neutrals so no new hue enters the set, and deliberately NOT blue --
# blue is reserved for present-weather.
MAGNITUDE_STOPS = ("#FFFFFF", "#D8D8D8", "#8C8C8C", "#1A1A1A")

# Neutrals
INK = "#1A1A1A"
MUTED = "#5A5A5A"
FAINT = "#B0B0B0"
GRID = "#DCDCDC"

# Component order -- Windows, Ground floor, Facade, Roof.  Fixed to match the
# state vector, Table 3 and the Fig 6 package names; Table 1 is the outlier.
COMPONENT_ORDER = ("windows", "floor", "facade", "roof")
COMPONENT_LABEL = {
    "windows": "Windows",
    "floor":   "Ground floor",
    "facade":  "Façade",
    "roof":    "Roof",
}


def tint(colour, amount=0.80):
    """Blend `colour` toward white by `amount` (0 = unchanged, 1 = white).

    Box diagrams need a pale fill behind black text.  Generators were inventing
    hex literals for this; routing it through here keeps every fill derivable
    from a namespaced hue instead of introducing a new one.
    """
    from matplotlib.colors import to_rgb
    if not 0.0 <= amount <= 1.0:
        raise ValueError(f"amount must be in [0, 1]; got {amount}")
    r, g, b = to_rgb(colour)
    return tuple(c + (1.0 - c) * amount for c in (r, g, b))


def magnitude_cmap(name="magnitude"):
    """Sequential colormap for a quantity.  Neutral by design: blue is reserved."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, list(MAGNITUDE_STOPS))


# ---------------------------------------------------------------------------
def apply(base_pt=8.0):
    """Install rcParams.  base_pt is the printed size of ordinary tick labels."""
    matplotlib.rcParams.update({
        # TrueType, not Type 3 -- the current figures embed DejaVu as Type 3,
        # which Elsevier's preflight flags.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        # ONE family for text and mathtext.  Arial body + DejaVu mathtext put
        # every $E_H$ in a different typeface from the words beside it.  DejaVu
        # ships with matplotlib, so the archive reproduces off this machine too.
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",

        "font.size": base_pt,
        "axes.labelsize": base_pt + 1,
        "axes.titlesize": base_pt + 1,
        "xtick.labelsize": base_pt,
        "ytick.labelsize": base_pt,
        "legend.fontsize": base_pt,
        "figure.titlesize": base_pt + 2,

        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,

        "lines.linewidth": 1.4,
        "lines.markersize": 5.0,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def size(frac, aspect=0.62):
    """figsize for a figure that will be included at `frac` of \\linewidth.

    Authoring at the final width is what makes nominal point sizes real: the
    LaTeX \\includegraphics scale becomes 1.0, so no downscaling occurs.
    `aspect` is height/width.
    """
    if not 0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1]; got {frac}")
    w = LINEWIDTH_IN * frac
    return (w, w * aspect)


def _rendered_glyph_sizes(path):
    """Every font size actually used in a written PDF, from its content streams.

    The artist-level check reads Text.get_fontsize(), which is the size of the
    whole label.  mathtext then renders sub/superscripts at a fraction of it, so
    the smallest thing on the page is routinely not any artist's fontsize.  This
    reads the `/F<n> <size> Tf` operators, i.e. what the renderer actually set.
    """
    import re, zlib
    with open(path, "rb") as fh:
        d = fh.read()
    sizes = set()
    for m in re.finditer(rb"stream\r?\n", d):
        start = m.end()
        end = d.find(b"endstream", start)
        try:
            chunk = zlib.decompress(d[start:end])
        except Exception:
            continue
        for t in re.finditer(rb"/[A-Za-z0-9_+-]+\s+([\d.]+)\s+Tf", chunk):
            sizes.add(round(float(t.group(1)), 2))
    if not sizes:
        # Every figure in this set carries text.  An empty result means the
        # streams did not parse, not that the page is clean -- passing here
        # would be a silent fallback that defeats the whole guard.
        raise ValueError(f"{path}: no text-setting operators found; the glyph "
                         f"guard could not read this PDF, so it cannot vouch "
                         f"for it. Do not treat this as a pass.")
    return sorted(sizes)


def _emitted_width_pt(path):
    """Width in points of a written PDF, read back from its MediaBox."""
    import re
    with open(path, "rb") as fh:
        d = fh.read()
    m = re.findall(rb"/MediaBox\s*\[([^\]]+)\]", d)
    if not m:
        raise ValueError(f"{path}: no MediaBox; cannot verify printed size")
    v = [float(x) for x in m[0].split()]
    return v[2] - v[0]


def audit(fig, frac, scale=1.0):
    """Return every text artist that would print below MIN_PT at `scale`."""
    offenders = []
    for t in fig.findobj(match=lambda o: hasattr(o, "get_fontsize")):
        try:
            pt = t.get_fontsize() * scale
        except Exception:
            continue
        s = (t.get_text() or "").strip() if hasattr(t, "get_text") else ""
        if s and pt < MIN_PT:
            offenders.append((round(pt, 2), s[:60]))
    return sorted(set(offenders))


def finalize(fig, stem, frac, formats=("pdf", "png"), strict=True,
             standalone=False):
    r"""Write `stem`.{pdf,png} only if the type survives both size checks.

    The contract, precisely: ORDINARY LETTERING >= MIN_PT (7.0); RENDERED MATH
    GLYPHS >= MIN_GLYPH_PT (6.0).  The two differ because mathtext sets
    sub/superscripts below their artist's nominal size, and journals accept a
    subscript smaller than body lettering.  Nothing here promises every mark on
    the page clears 7 pt.

    Output is staged beside the destination and moved into place only after both
    checks pass.  A rejected generation therefore publishes no new output, leaves
    any already-accepted destination file untouched, and removes every staging
    file -- including when the failure comes from the PDF parse rather than from
    the type itself.

    The check is done against the size actually PRINTED, not the authored size:
    savefig.bbox="tight" crops the canvas, so LaTeX rescales the emitted file to
    reach frac*\linewidth and nominal pt != printed pt.  The emitted width is
    read back from the PDF and the true scale computed from it.  `standalone`
    skips the \linewidth rule for figures placed at a fixed physical size
    (the journal graphical abstract).
    """
    import os
    if "pdf" not in formats:
        raise ValueError("finalize needs 'pdf' in formats: both guards measure "
                         "the emitted PDF, not the in-memory figure.")

    # Stage in the destination directory so the final move is a same-filesystem
    # atomic rename rather than a copy that can be interrupted half-written.
    d = os.path.dirname(os.path.abspath(stem)) or "."
    base = os.path.basename(stem)
    staged = {ext: os.path.join(d, f".{base}.staging.{ext}") for ext in formats}

    def _discard():
        for q in staged.values():
            try:
                os.unlink(q)
            except FileNotFoundError:
                pass

    # ONE cleanup point: any failure between here and publication -- savefig,
    # MediaBox read, artist audit, glyph parse, or either floor -- must leave no
    # staging file and must not disturb an already-accepted destination file.
    try:
        for ext in formats:
            fig.savefig(staged[ext], dpi=600 if ext == "png" else 300)

        scale = 1.0
        if not standalone:
            emitted = _emitted_width_pt(staged["pdf"])
            target = frac * LINEWIDTH_IN * 72.0
            scale = target / emitted
            print(f"  {stem.split('/')[-1]}: emitted {emitted:.1f} pt -> placed at "
                  f"{target:.1f} pt (scale {scale:.3f})")

        # Artist level: ordinary lettering against MIN_PT.
        bad = audit(fig, frac, scale=scale)

        # Renderer level: what the PDF actually sets, against MIN_GLYPH_PT.
        # Catches the mathtext sub/superscript shrink the artist check cannot see.
        glyphs = [g * scale for g in _rendered_glyph_sizes(staged["pdf"])]
        tiny = [round(g, 2) for g in glyphs if g < MIN_GLYPH_PT]
        if tiny:
            report = (f"{stem}: rendered glyph(s) at {tiny} pt, below "
                      f"MIN_GLYPH_PT={MIN_GLYPH_PT} (printed scale {scale:.3f}). "
                      f"Usually a nested mathtext subscript such as $D_{{T_u}}$ -- "
                      f"flatten it rather than enlarging the whole label.")
            if strict:
                raise ValueError(report)
            print("WARNING " + report)
        elif glyphs:
            print(f"    smallest rendered glyph {min(glyphs):.2f} pt")

        if bad:
            msg = "\n".join(f"    {pt} pt  {txt!r}" for pt, txt in bad)
            report = (f"{stem}: {len(bad)} text item(s) below {MIN_PT} pt at "
                      f"frac={frac} (printed scale {scale:.3f}):\n{msg}")
            if strict:
                raise ValueError(report)
            print("WARNING " + report)
    except BaseException:
        _discard()
        raise

    for ext in formats:                      # both checks passed -- publish
        os.replace(staged[ext], f"{stem}.{ext}")
    plt.close(fig)
    return f"{stem}.pdf"
