"""
Unified visualization theme setup for excog-trajectory.

Call apply_theme() once at program start (CLI entry) to ensure consistent
styling across figures without repeating style parameters.
"""
from __future__ import annotations

from typing import Optional

import matplotlib as mpl
import seaborn as sns


_DEFAULT_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def apply_theme(context: str = "talk", style: str = "whitegrid", rc: Optional[dict] = None) -> None:
    """Apply a consistent plotting theme across the project.

    Parameters
    ----------
    context : str
        Seaborn context (e.g., "paper", "notebook", "talk", "poster").
    style : str
        Seaborn style (e.g., "white", "whitegrid", "ticks").
    rc : Optional[dict]
        Additional rcParams to update.
    """
    sns.set_theme(context=context, style=style)
    rc_params = dict(_DEFAULT_RC)
    if rc:
        rc_params.update(rc)
    mpl.rcParams.update(rc_params)


__all__ = ["apply_theme"]
