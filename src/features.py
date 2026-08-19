"""
Feature construction and chronological splitting (Chapter 3.5).

Demand model : window of recent readings + calendar information
Solar model  : calendar + weather variables + installed capacity

Calendar features are encoded cyclically. Raw integer hour is never used:
hour 23 and hour 0 must be adjacent in feature space.

Splitting is strictly chronological. Shuffling would give the model a glimpse
of the future during training and render the predictions meaningless (3.5).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C

WEATHER_COLS = ["temperature_2m", "shortwave_radiation", "cloud_cover",
                "wind_speed_10m", "relative_humidity_2m"]


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    sp = C.STEPS_PER_DAY
    step = idx.hour * (60 // C.FREQ_MIN) + idx.minute // C.FREQ_MIN
    return pd.DataFrame({
        "tod_sin": np.sin(2 * np.pi * step / sp),
        "tod_cos": np.cos(2 * np.pi * step / sp),
        "dow_sin": np.sin(2 * np.pi * idx.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * idx.dayofweek / 7),
        "doy_sin": np.sin(2 * np.pi * idx.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * idx.dayofyear / 365.25),
        "is_weekend": (idx.dayofweek >= 5).astype(float),
    }, index=idx)


def demand_features(y: pd.Series, lags=(1, 2, 3, 48, 96, 336)) -> pd.DataFrame:
    """Recent readings + calendar. All lags are strictly backward-looking."""
    X = calendar_features(y.index)
    for L in lags:
        X[f"lag_{L}"] = y.shift(L)
    X["roll_mean_48"] = y.shift(1).rolling(48).mean()
    X["roll_std_48"] = y.shift(1).rolling(48).std()
    X["roll_mean_336"] = y.shift(1).rolling(336).mean()
    return X


def solar_features(y: pd.Series, weather: pd.DataFrame, kwp: float,
                   lags=(1, 2, 48)) -> pd.DataFrame:
    """Calendar + weather + installed capacity (3.4.2)."""
    X = calendar_features(y.index)
    for c in WEATHER_COLS:
        if c in weather.columns:
            X[c] = weather[c].reindex(y.index).values
    if "shortwave_radiation" in X.columns:
        X["swr_lag1"] = X["shortwave_radiation"].shift(1)
        X["swr_roll3"] = X["shortwave_radiation"].rolling(3).mean()
    for L in lags:
        X[f"lag_{L}"] = y.shift(L)
    X["kwp"] = kwp
    return X


def align(X: pd.DataFrame, y: pd.Series):
    m = X.notna().all(axis=1) & y.notna()
    return X[m], y[m]


def chrono_split(X: pd.DataFrame, y: pd.Series,
                 train=C.TRAIN_FRAC, val=C.VAL_FRAC):
    """Chronological train / validation / test split (3.5)."""
    n = len(X)
    i1, i2 = int(n * train), int(n * (train + val))
    return ((X.iloc[:i1], y.iloc[:i1]),
            (X.iloc[i1:i2], y.iloc[i1:i2]),
            (X.iloc[i2:], y.iloc[i2:]))


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len=C.SEQ_LEN):
    """(N, seq_len, n_features) windows predicting the next value."""
    n = len(X) - seq_len
    if n <= 0:
        raise ValueError("series shorter than seq_len")
    Xs = np.stack([X[i:i + seq_len] for i in range(n)]).astype(np.float32)
    return Xs, y[seq_len:].astype(np.float32)
