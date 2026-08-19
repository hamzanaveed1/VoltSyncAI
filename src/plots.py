"""
Figures (Chapter 3.7): predicted vs actual on sample days, plus error summaries.
Visual inspection can reveal problems that summary numbers hide.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C

TEAL, AMBER, GREY, RED = "#0d7a6f", "#f5a623", "#6b7a82", "#c0392b"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25,
                     "figure.autolayout": True})


def _save(fig, name):
    p = C.FIGS / name
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def pred_vs_actual(pred, unit, task, days=3, fignum="4.1"):
    n = days * C.STEPS_PER_DAY
    d = pred[unit]
    t, a, p, b = d["time"][:n], d["actual"][:n], d["pred"][:n], d["base"][:n]
    col = TEAL if task == "demand" else AMBER
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t, a, color="black", lw=1.5, label="Actual")
    ax.plot(t, p, color=col, lw=1.4, label="Model forecast")
    ax.plot(t, b, color=GREY, lw=1.0, ls="--", alpha=.85, label="Persistence baseline")
    ax.set_ylabel("kWh / 30 min")
    ax.set_title(f"Figure {fignum}  {task.capitalize()} forecast vs actual — {unit}")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    return _save(fig, f"fig{fignum.replace('.','_')}_{task}_pred_vs_actual.png")


def error_bars(df, task, metric, fignum="4.3"):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(df)); w = .38
    ax.bar(x - w/2, df[metric], w, color=TEAL if task=="demand" else AMBER,
           label="Model")
    ax.bar(x + w/2, df["baseline_" + metric.replace("_daylight","")], w,
           color=GREY, label="Persistence baseline")
    ax.set_xticks(x); ax.set_xticklabels(df["unit"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(metric.upper().replace("_", " "))
    ax.set_title(f"Figure {fignum}  {task.capitalize()} — {metric.upper()} by unit")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    return _save(fig, f"fig{fignum.replace('.','_')}_{task}_{metric}.png")


def learning_curve(hist, unit, fignum="4.5"):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(hist["loss"], color=TEAL, label="Training loss")
    if "val_loss" in hist:
        ax.plot(hist["val_loss"], color=AMBER, label="Validation loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (scaled)")
    ax.set_title(f"Figure {fignum}  LSTM learning curve — {unit}")
    ax.legend(frameon=False)
    return _save(fig, f"fig{fignum.replace('.','_')}_lstm_learning_curve.png")


def feature_importance(imp, ss, fignum="4.6"):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    imp.sort_values().plot(kind="barh", ax=ax, color=AMBER)
    ax.set_xlabel("Importance"); ax.set_ylabel("")
    ax.set_title(f"Figure {fignum}  Gradient boosting feature importance — ss_{ss}")
    return _save(fig, f"fig{fignum.replace('.','_')}_gbm_importance.png")


def daily_profile(pred, unit, task, fignum="4.7"):
    d = pred[unit]
    df = pd.DataFrame({"a": d["actual"], "p": d["pred"]},
                      index=pd.DatetimeIndex(d["time"]))
    g = df.groupby(df.index.hour).mean()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(g.index, g["a"], color="black", lw=1.6, label="Actual")
    ax.plot(g.index, g["p"], color=TEAL if task=="demand" else AMBER,
            lw=1.6, ls="--", label="Forecast")
    ax.set_xlabel("Hour (UTC)"); ax.set_ylabel("Mean kWh / 30 min")
    ax.set_title(f"Figure {fignum}  Mean daily profile — {unit}")
    ax.legend(frameon=False)
    return _save(fig, f"fig{fignum.replace('.','_')}_{task}_daily_profile.png")
