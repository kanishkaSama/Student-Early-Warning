"""Shared chart style so every figure in the report looks consistent."""
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"     # "not at risk" / single series
ORANGE = "#eb6834"   # "at risk"
OUTCOME_COLORS = {"Not at risk": BLUE, "At risk": ORANGE}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "figure.dpi": 110, "savefig.dpi": 150,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY, "text.color": INK, "font.size": 10,
        "legend.frameon": False, "axes.axisbelow": True, "lines.linewidth": 2,
    })


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
