"""
Baseline methods (Chapter 3.4.3).

Seasonal persistence: the forecast for any half hour is the value observed at
the same time on the previous day. Deliberately simple, and surprisingly hard
to beat where the daily pattern is strong. Included to guard against the common
error of declaring a complex model successful when it has not actually improved
on a trivial one.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C


def seasonal_persistence(y: pd.Series, period=C.STEPS_PER_DAY) -> pd.Series:
    """Value at the same time yesterday."""
    return y.shift(period)


def naive_persistence(y: pd.Series) -> pd.Series:
    """Previous half hour. Reported as a secondary reference only."""
    return y.shift(1)
