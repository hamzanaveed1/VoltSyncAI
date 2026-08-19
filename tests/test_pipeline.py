"""
Correctness tests for the VoltSync AI forecasting pipeline.

These guard the properties that are easy to break silently and expensive to
discover late: data leakage, split ordering, metric definitions, and the
real/synthetic mode boundary.

    python -m pytest tests/ -q      (or)      python tests/test_pipeline.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config as C
from src.data import build_synthetic, select_ideal_homes, load_data
from src.features import (demand_features, solar_features, align, chrono_split,
                          make_sequences, calendar_features)
from src.baselines import seasonal_persistence
from src.metrics import mape, daylight_mape, mae, rmse, report
from src.weather import grid_key, align_to_index, synth_weather
from src.model_lstm import Scaler

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILED.append(name)


def test_splits_are_chronological():
    pv, dem, meta = build_synthetic(n_pv=1, n_homes=1)
    y = dem.iloc[:, 0]
    X = demand_features(y); X, y = align(X, y)
    (Xtr, _), (Xva, _), (Xte, _) = chrono_split(X, y)
    check("splits are chronological, no overlap",
          Xtr.index.max() < Xva.index.min() < Xva.index.max() < Xte.index.min())
    check("split proportions respected",
          abs(len(Xtr) / len(X) - C.TRAIN_FRAC) < 0.01,
          f"{len(Xtr)/len(X):.3f} vs {C.TRAIN_FRAC}")


def test_no_target_leakage_in_features():
    pv, dem, meta = build_synthetic(n_pv=1, n_homes=1)
    y = dem.iloc[:, 0]
    X = demand_features(y); X, y = align(X, y)
    # No feature may equal the contemporaneous target
    worst = max(abs(np.corrcoef(X[c].values, y.values)[0, 1]) for c in X.columns)
    check("no feature is the target itself", worst < 0.999, f"max |r| = {worst:.4f}")
    # lag_1 must equal y shifted by one step
    lag1 = y.shift(1).reindex(X.index)
    ok = np.allclose(X["lag_1"].values[1:], lag1.values[1:], equal_nan=True)
    check("lag features are strictly backward-looking", ok)


def test_scaler_fitted_on_train_only():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(0, 1, (100, 5, 3)); Xte = rng.normal(50, 1, (20, 5, 3))
    sc = Scaler().fit(Xtr)
    mu_before = sc.mu.copy()
    sc.transform(Xte)
    check("transform() does not refit the scaler", np.allclose(mu_before, sc.mu))
    check("train statistics are not contaminated by test", abs(sc.mu.mean()) < 0.5,
          f"mu={sc.mu.mean():.3f}")


def test_mape_definitions():
    y = np.array([0.0, 0.0, 2.0, 4.0]); yh = np.array([0.1, 0.0, 2.2, 3.6])
    check("plain MAPE skips exact zeros", np.isfinite(mape(y, yh)))
    dm = daylight_mape(y, yh, C.DAYLIGHT_MIN_KWH)
    check("daylight MAPE excludes sub-threshold values", abs(dm - 10.0) < 1e-6,
          f"{dm:.4f}%")
    check("all-zero series yields NaN, not inf",
          np.isnan(daylight_mape(np.zeros(5), np.zeros(5), 0.05)))


def test_baseline_is_yesterday():
    idx = pd.date_range("2022-01-01", periods=200, freq="30min", tz="UTC")
    y = pd.Series(np.arange(200.0), index=idx)
    b = seasonal_persistence(y)
    check("seasonal persistence lags exactly one day",
          b.iloc[C.STEPS_PER_DAY] == y.iloc[0])


def test_period_ending_shift():
    hourly = pd.DataFrame(
        {"shortwave_radiation": np.arange(48.0)},
        index=pd.date_range("2022-06-01", periods=48, freq="h", tz="UTC"))
    idx = pd.date_range("2022-06-01", periods=48, freq="30min", tz="UTC")
    pe = align_to_index(hourly, idx, period_ending=True)
    inst = align_to_index(hourly, idx, period_ending=False)
    diff = float((inst - pe).dropna().iloc[:, 0].mean())
    check("period-ending sampling shifts weather by 15 min",
          abs(diff - 0.25) < 1e-6, f"{diff:.4f} h of an hourly ramp")


def test_weather_grid_dedup():
    check("nearby coords collapse to one grid cell",
          grid_key(53.38, -1.47) == grid_key(53.41, -1.52))
    check("distant coords do not collapse",
          grid_key(53.38, -1.47) != grid_key(51.50, -0.12))


def test_real_mode_refuses_silent_fallback():
    old = C.DATA_MODE
    C.DATA_MODE = "real"
    try:
        load_data(n_pv=1, n_homes=1)
        ok = False
    except ValueError:
        ok = True
    finally:
        C.DATA_MODE = old
    check("DATA_MODE='real' refuses to fall back to synthetic", ok)


def test_home_selection_by_coverage():
    idx = pd.date_range(C.DEMAND_START, C.DEMAND_END, freq="30min", tz="UTC",
                        inclusive="left")
    good = pd.Series(1.0, index=idx)
    poor = pd.Series(1.0, index=idx).mask(np.arange(len(idx)) > len(idx) * 0.3)
    sel = select_ideal_homes({"good": good, "poor": poor}, n=1)
    check("home selection ranks by record completeness", sel == ["good"])


def test_sequences_shape_and_alignment():
    X = np.arange(100 * 3, dtype=float).reshape(100, 3)
    y = np.arange(100, dtype=float)
    Xs, ys = make_sequences(X, y, seq_len=10)
    check("sequence windows have shape (N, seq_len, features)",
          Xs.shape == (90, 10, 3), str(Xs.shape))
    check("target aligns to the step after each window", ys[0] == y[10])


def test_cyclical_encoding_wraps():
    idx = pd.date_range("2022-01-01", periods=48, freq="30min", tz="UTC")
    c = calendar_features(idx)
    d = np.hypot(c["tod_sin"].iloc[0] - c["tod_sin"].iloc[-1],
                 c["tod_cos"].iloc[0] - c["tod_cos"].iloc[-1])
    check("hour 23:30 is adjacent to 00:00 in feature space", d < 0.15,
          f"distance {d:.4f}")


def test_windows_match_dataset_coverage():
    """IDEAL ends June 2018; uk_pv runs to 2025. The windows must not be forced
    to coincide, or zero IDEAL homes survive selection."""
    d0, d1 = pd.Timestamp(C.DEMAND_START), pd.Timestamp(C.DEMAND_END)
    check("demand window sits inside IDEAL coverage (2016-09 .. 2018-06)",
          d0 >= pd.Timestamp("2016-08-01") and d1 <= pd.Timestamp("2018-07-01"),
          f"{C.DEMAND_START}..{C.DEMAND_END}")
    p0, p1 = pd.Timestamp(C.PV_START), pd.Timestamp(C.PV_END)
    check("PV window sits inside uk_pv coverage (2010 .. 2025)",
          p0 >= pd.Timestamp("2010-01-01") and p1 <= pd.Timestamp("2025-12-31"),
          f"{C.PV_START}..{C.PV_END}")
    check("windows are independent (tasks are never joined)",
          (C.DEMAND_START, C.DEMAND_END) != (C.PV_START, C.PV_END))


def test_synthetic_physics():
    pv, dem, meta = build_synthetic(n_pv=3, n_homes=3)
    pv["kwh"] = pv.generation_Wh / 1000
    night = pv[(pv.datetime_GMT.dt.hour <= 2) | (pv.datetime_GMT.dt.hour >= 23)]
    check("no night-time solar generation", night.kwh.sum() == 0.0)
    y22 = pv[pv.datetime_GMT.dt.year == 2022].groupby("ss_id").kwh.sum()
    sy = (y22 / meta.set_index("ss_id").kWp)
    check("specific yield within UK range (750-950 kWh/kWp)",
          750 <= sy.mean() <= 950, f"{sy.mean():.0f}")
    check("demand is strictly positive", (dem.values > 0).all())


if __name__ == "__main__":
    print("VoltSync AI - pipeline tests\n" + "-" * 52)
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print("-" * 52)
    print(f"{'ALL TESTS PASSED' if not FAILED else 'FAILURES: ' + ', '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)
