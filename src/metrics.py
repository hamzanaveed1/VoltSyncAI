"""
Evaluation metrics (Chapter 3.7).

MAE  - average error size in kWh; simple to interpret
RMSE - penalises occasional large misses
MAPE - error as a percentage; comparable across households of different size

Solar MAPE is computed over DAYTIME ONLY. Night generation is exactly zero, so
percentage error is undefined there. This is stated in Table 3.2 and is applied
consistently throughout.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-9
_a = lambda x: np.asarray(x, dtype=float).ravel()


def mae(y, yhat):  return float(np.mean(np.abs(_a(y) - _a(yhat))))
def rmse(y, yhat): return float(np.sqrt(np.mean((_a(y) - _a(yhat)) ** 2)))


def mape(y, yhat):
    """Percentage error. Valid for demand, which is never zero."""
    y, yhat = _a(y), _a(yhat)
    m = np.abs(y) > EPS
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m])) * 100) if m.any() else np.nan


def daylight_mape(y, yhat, threshold):
    """Solar MAPE restricted to daytime steps (Table 3.2)."""
    y, yhat = _a(y), _a(yhat)
    m = y > threshold
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m])) * 100) if m.any() else np.nan


def daylight_fraction(y, threshold):
    return float((_a(y) > threshold).mean())


def skill(y, yhat, y_base):
    """Improvement over the baseline in MAE terms. >0 means the model helps."""
    b = mae(y, y_base)
    return float((b - mae(y, yhat)) / b * 100) if b > EPS else np.nan


def report(y, yhat, y_base=None, task="demand", threshold=None) -> dict:
    out = {"mae": mae(y, yhat), "rmse": rmse(y, yhat), "n": int(_a(y).size)}
    if task == "demand":
        out["mape"] = mape(y, yhat)
    else:
        thr = threshold if threshold is not None else 0.05
        out["mape_daylight"] = daylight_mape(y, yhat, thr)
        out["daylight_fraction"] = daylight_fraction(y, thr)
    if y_base is not None:
        out["baseline_mae"] = mae(y, y_base)
        out["baseline_rmse"] = rmse(y, y_base)
        out["baseline_mape"] = (mape(y, y_base) if task == "demand"
                                else daylight_mape(y, y_base, threshold or 0.05))
        out["mae_improvement_pct"] = skill(y, yhat, y_base)
        out["beats_baseline"] = bool(out["mae_improvement_pct"] > 0)
    return out
