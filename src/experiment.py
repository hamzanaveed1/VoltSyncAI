"""
The forecasting experiment (Chapters 1.6 and 3.7).

RQ1: How accurately can an LSTM forecast short-term household demand compared
     with a simple baseline?
RQ2: How accurately can a gradient boosting model forecast rooftop solar
     generation compared with a simple baseline?

Results are reported per household / per system AND averaged, always alongside
the same measures for the baseline (3.7). A model is judged useful only where
it is clearly better than its baseline.
"""
from __future__ import annotations
import json
import pickle
import time
import numpy as np
import pandas as pd

from . import config as C
from .data import load_data
from .features import (demand_features, solar_features, align, chrono_split,
                       make_sequences)
from .model_lstm import DemandLSTM
from .model_gbm import SolarGBM
from .baselines import seasonal_persistence, naive_persistence
from .metrics import report
from .weather import get_weather, get_weather_at


# ---------------------------------------------------------------- RQ1
def run_demand(demand: pd.DataFrame, verbose=True):
    rows, preds, histories = [], {}, {}
    part = C.RESULTS / "partial"
    part.mkdir(exist_ok=True)
    for home in demand.columns:
        cache = part / f"demand_{home}.pkl"
        if cache.exists():                      # resume across sessions
            d = pickle.loads(cache.read_bytes())
            rows.append(d["row"]); preds[home] = d["pred"]; histories[home] = d["hist"]
            if verbose:
                print(f"    {home:10} [cached]  MAE {d['row']['mae']:.4f}  "
                      f"MAPE {d['row']['mape']:5.2f}%  "
                      f"-> {d['row']['mae_improvement_pct']:+.1f}%")
            continue
        y = demand[home].dropna()
        X = demand_features(y)
        X, y = align(X, y)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = chrono_split(X, y)

        Xs_tr, ys_tr = make_sequences(Xtr.values, ytr.values)
        Xs_va, ys_va = make_sequences(Xva.values, yva.values)
        Xs_te, ys_te = make_sequences(Xte.values, yte.values)
        t_idx = Xte.index[C.SEQ_LEN:]

        t0 = time.time()
        m = DemandLSTM().fit(Xs_tr, ys_tr, Xs_va, ys_va)
        yhat = m.predict(Xs_te)
        train_s = time.time() - t0

        base = seasonal_persistence(y).reindex(t_idx).values
        ok = ~np.isnan(base)
        r = report(ys_te[ok], yhat[ok], base[ok], task="demand")
        r.update(unit=home, task="demand", model="LSTM",
                 baseline="Seasonal persistence (t-24h)",
                 epochs_run=len(m.history["loss"]), train_seconds=round(train_s, 1))
        rows.append(r)
        preds[home] = {"time": t_idx[ok], "actual": ys_te[ok],
                       "pred": yhat[ok], "base": base[ok]}
        histories[home] = m.history
        if verbose:
            print(f"    {home:10} MAE {r['mae']:.4f}  RMSE {r['rmse']:.4f}  "
                  f"MAPE {r['mape']:5.2f}%  | baseline MAE {r['baseline_mae']:.4f}  "
                  f"-> {r['mae_improvement_pct']:+.1f}%")
        m.save(f"lstm_{home}")
        cache.write_bytes(pickle.dumps({"row": r, "pred": preds[home],
                                        "hist": m.history}))
    return pd.DataFrame(rows), preds, histories


# ---------------------------------------------------------------- RQ2
def run_solar(pv: pd.DataFrame, meta: pd.DataFrame, weather=None,
              verbose=True):
    """
    Weather is fetched for the approximate location of EACH system (3.3.3),
    not once for the region. Requests are memoised per reanalysis grid cell,
    so co-located systems cost a single call.
    """
    rows, preds, importances = [], {}, {}
    for _, mrow in meta.iterrows():
        ss = int(mrow["ss_id"])
        s = pv[pv.ss_id == ss].set_index("datetime_GMT")["generation_Wh"] / 1000.0
        s = s.sort_index().asfreq(f"{C.FREQ_MIN}min")
        w = get_weather_at(float(mrow["latitude_rounded"]),
                           float(mrow["longitude_rounded"]), s.index)
        X = solar_features(s, w, float(mrow["kWp"]))
        X, y = align(X, s)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = chrono_split(X, y)

        t0 = time.time()
        m = SolarGBM().fit(Xtr, ytr, Xva, yva)
        yhat = m.predict(Xte)
        train_s = time.time() - t0

        base = seasonal_persistence(y).reindex(Xte.index).values
        ok = ~np.isnan(base)
        r = report(yte.values[ok], yhat[ok], base[ok], task="solar",
                   threshold=C.DAYLIGHT_MIN_KWH)
        r.update(unit=f"ss_{ss}", task="solar", model="Gradient boosting",
                 baseline="Seasonal persistence (t-24h)", kwp=float(mrow["kWp"]),
                 best_iteration=m.best_iteration, train_seconds=round(train_s, 1))
        rows.append(r)
        preds[ss] = {"time": Xte.index[ok], "actual": yte.values[ok],
                     "pred": yhat[ok], "base": base[ok]}
        importances[ss] = m.importance()
        if verbose:
            print(f"    ss_{ss}   MAE {r['mae']:.4f}  RMSE {r['rmse']:.4f}  "
                  f"MAPE(day) {r['mape_daylight']:5.2f}%  | baseline MAE "
                  f"{r['baseline_mae']:.4f}  -> {r['mae_improvement_pct']:+.1f}%")
        m.save(f"gbm_ss{ss}")
    return pd.DataFrame(rows), preds, importances


def summarise(df: pd.DataFrame, task: str) -> pd.Series:
    """Mean across units, as required by 3.7."""
    cols = ["mae", "rmse", "baseline_mae", "baseline_rmse",
            "mae_improvement_pct"]
    cols += ["mape", "baseline_mape"] if task == "demand" else \
            ["mape_daylight", "baseline_mape"]
    s = df[[c for c in cols if c in df.columns]].mean()
    s["units"] = len(df)
    s["units_beating_baseline"] = int(df["beats_baseline"].sum())
    return s
