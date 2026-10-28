"""Cohere / Cohere Labs figure style (research/10_paper_playbook.md §3).

Palette: cohere.com production CSS tokens and colours sampled from the Tiny Aya report's own charts (arXiv 2603.11510).
Validated with the dataviz checker (blue, green, coral-700, quartz-700: all CVD checks pass; green is 2.2:1 on white,
so green marks always carry a direct label). Type: Space Grotesk (labels) and Space Mono (numbers), the OFL fonts
nearest Cohere's CohereText / CohereMono, vendored in paper/fonts/.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from common import ROOT

INK, AXIS, GRID, SURFACE = "#16270d", "#8a9386", "#e3e6e1", "#ffffff"
BLUE, BLUE_L, GREEN, GREEN_D = "#4c6ee6", "#8fa6f9", "#5bbf8a", "#357a4d"
CORAL, CORAL_L, QUARTZ, MUSH = "#ca492d", "#ff7759", "#9b60aa", "#afa694"
SANS, MONO = "Space Grotesk", "Space Mono"
W = 5.5  # NeurIPS text width, inches


def apply():
    for f in (ROOT / "paper" / "fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update(
        {
            "font.family": [SANS, "DejaVu Sans"],  # per-glyph fallback: Space Grotesk has no Greek (ε)
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "axes.titleweight": "medium",
            "axes.titlelocation": "left",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.pad": 4,
            "ytick.major.pad": 4,
            "legend.frameon": False,
            "legend.fontsize": 6.8,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "pdf.fonttype": 42,
            "figure.dpi": 200,
        }
    )


def mono_ticks(ax, axes="xy"):
    """Cohere Labs charts set numbers in the mono face."""
    for which in axes:
        for t in ax.get_xticklabels() if which == "x" else ax.get_yticklabels():
            t.set_fontfamily(MONO)
            t.set_fontsize(6.6)


def label(ax, x, y, text, color=INK, **kw):
    """Direct data label in mono (the green series has to carry one: 2.2:1 contrast on white)."""
    ax.annotate(text, (x, y), fontfamily=MONO, fontsize=6.2, color=color, **kw)


def action_title(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=8.4, fontweight="bold", color=INK)
