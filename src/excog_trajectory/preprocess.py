"""
Preprocessing utilities to centralize scaling and avoid leakage.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def standardize_targets(y: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on y and return transformed y and the scaler.

    This helper centralizes scaling to reduce ad-hoc usage and ensures
    consistent behavior across modules.
    """
    scaler = StandardScaler().fit(y)
    y_std = pd.DataFrame(scaler.transform(y), index=y.index, columns=y.columns)
    return y_std, scaler


__all__ = ["standardize_targets"]
