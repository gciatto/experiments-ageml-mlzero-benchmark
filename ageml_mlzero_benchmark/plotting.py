"""Shared plot appearance, matching AGE-ML campaign analysis figures."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from textwrap import fill

from matplotlib import RcParams
from matplotlib.figure import Figure

# Same fonts, sizes and vector-font settings as AGE-ML's campaign_analysis.py.
STYLE = RcParams(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "axes.titleweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)
COLORS = {
    "ageml": "#345995",
    "mlzero": "#479f76",
    "repair": "#8c6bb1",
    "timeout": "#d79b35",
    "setup": "#c65f53",
    "neutral": "#66717e",
    "grid": "#e5e7eb",
}


@contextmanager
def plot_style() -> Iterator[None]:
    """Keep report styling local instead of changing the caller's rcParams."""
    import matplotlib

    with matplotlib.rc_context(STYLE):
        yield


def dataset_label(key: str) -> str:
    return {"har": "HAR", "mnist_784": "MNIST", "auto_mpg": "Auto MPG"}.get(
        key, key.replace("_", " ").title()
    )


def save_figure(
    figure: Figure,
    directory: Path,
    name: str,
    *,
    formats: Sequence[str] = ("png", "pdf", "svg"),
    footnote: str | None = None,
    layout: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    if footnote:
        figure.text(
            0.01,
            0.015,
            fill(footnote, width=max(40, int(figure.get_figwidth() * 15))),
            fontsize=8,
            color="#555555",
            va="bottom",
        )
    if layout:
        figure.tight_layout(rect=(0, 0.23 if footnote else 0, 1, 0.98))
    for extension in formats:
        figure.savefig(directory / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(figure)
