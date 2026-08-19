#!/usr/bin/env python3
"""
VoltSync AI - An AI Forecasting Engine for Residential Demand & Solar Generation
Main experiment runner.

    python run_experiment.py [--homes N] [--systems N]

Answers RQ1 (LSTM vs baseline, demand) and RQ2 (GBM vs baseline, solar).
Writes results/, figs/ and models/.
"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from src import config as C
from src.data import load_data, discover_ideal_files
from src.weather import get_weather
from src.experiment import run_demand, run_solar, summarise
from src import plots as PL


def banner(s):
    print(f"\n{'='*70}\n{s}\n{'='*70}")


def main(n_homes, n_systems, pv_path=None, ideal_files=None):
    t0 = time.time()
    banner("VoltSync AI - Forecasting Engine")
    print("  CRISP-DM phases (3.8): problem understanding -> data understanding")
    print("    -> data preparation -> modelling -> evaluation -> [deployment: "
          "out of scope, 1.8]")
    print(f"  mode={C.DATA_MODE}  window {C.START} to {C.END}  "
          f"{C.FREQ_MIN}-min steps")
    print(f"  horizon: next {C.FREQ_MIN} min   LSTM lookback: {C.SEQ_LEN} steps "
          f"({C.SEQ_LEN*C.FREQ_MIN/60:.0f} h)")

    # ------------------------------------------------------------ data
    banner("Objective ii - Data preparation")
    pv, demand, meta = load_data(n_pv=n_systems, n_homes=n_homes,
                                 pv_path=pv_path, ideal_files=ideal_files)
    weather = get_weather(demand.index)
    for name, log in [("solar", pv.attrs.get("cleaning_log"))]:
        if log:
            print(f"  {name} cleaning log:")
            for step, k in log:
                print(f"    {step:45} {k:>10,}")
    print(f"  demand : {demand.shape[0]:,} steps x {demand.shape[1]} homes")
    print(f"  solar  : {len(pv):,} rows x {pv.ss_id.nunique()} systems")
    print(f"  weather: {list(weather.columns)}  (per-system for solar)")
    split = f"{C.TRAIN_FRAC:.0%} train / {C.VAL_FRAC:.0%} val / {C.TEST_FRAC:.0%} test"
    print(f"  split  : {split}, chronological (never shuffled)")
    meta.to_csv(C.RESULTS / "table_3_1_pv_systems.csv", index=False)

    # ------------------------------------------------------------ RQ1
    banner("Objective iii+iv - RQ1: LSTM demand forecasting")
    dem_res, dem_pred, dem_hist = run_demand(demand)
    dem_res.to_csv(C.RESULTS / "table_4_1_demand_results.csv", index=False)
    s1 = summarise(dem_res, "demand")
    print(f"\n  MEAN across {int(s1['units'])} homes:")
    print(f"    LSTM      MAE {s1['mae']:.4f}  RMSE {s1['rmse']:.4f}  "
          f"MAPE {s1['mape']:.2f}%")
    print(f"    Baseline  MAE {s1['baseline_mae']:.4f}  "
          f"RMSE {s1['baseline_rmse']:.4f}  MAPE {s1['baseline_mape']:.2f}%")
    print(f"    Improvement in MAE: {s1['mae_improvement_pct']:+.1f}%   "
          f"homes beating baseline: {int(s1['units_beating_baseline'])}/{int(s1['units'])}")

    # ------------------------------------------------------------ RQ2
    banner("Objective iii+iv - RQ2: Gradient boosting solar forecasting")
    sol_res, sol_pred, sol_imp = run_solar(pv, meta)   # per-system weather
    sol_res.to_csv(C.RESULTS / "table_4_2_solar_results.csv", index=False)
    s2 = summarise(sol_res, "solar")
    print(f"\n  MEAN across {int(s2['units'])} systems:")
    print(f"    GBM       MAE {s2['mae']:.4f}  RMSE {s2['rmse']:.4f}  "
          f"MAPE(day) {s2['mape_daylight']:.2f}%")
    print(f"    Baseline  MAE {s2['baseline_mae']:.4f}  "
          f"RMSE {s2['baseline_rmse']:.4f}  MAPE(day) {s2['baseline_mape']:.2f}%")
    print(f"    Improvement in MAE: {s2['mae_improvement_pct']:+.1f}%   "
          f"systems beating baseline: {int(s2['units_beating_baseline'])}/{int(s2['units'])}")

    # ------------------------------------------------------------ summary
    summary = pd.DataFrame([
        {"task": "Demand (RQ1)", "model": "LSTM",
         "mae": s1["mae"], "rmse": s1["rmse"], "mape": s1["mape"],
         "baseline_mae": s1["baseline_mae"], "baseline_rmse": s1["baseline_rmse"],
         "baseline_mape": s1["baseline_mape"],
         "mae_improvement_pct": s1["mae_improvement_pct"]},
        {"task": "Solar (RQ2)", "model": "Gradient boosting",
         "mae": s2["mae"], "rmse": s2["rmse"], "mape": s2["mape_daylight"],
         "baseline_mae": s2["baseline_mae"], "baseline_rmse": s2["baseline_rmse"],
         "baseline_mape": s2["baseline_mape"],
         "mae_improvement_pct": s2["mae_improvement_pct"]},
    ])
    summary.to_csv(C.RESULTS / "table_4_3_summary.csv", index=False)

    # ------------------------------------------------------------ figures
    banner("Figures")
    h0 = demand.columns[0]; ss0 = int(meta.ss_id.iloc[0])
    figs = [
        PL.pred_vs_actual(dem_pred, h0, "demand", fignum="4.1"),
        PL.pred_vs_actual(sol_pred, ss0, "solar", fignum="4.2"),
        PL.error_bars(dem_res, "demand", "mae", fignum="4.3"),
        PL.error_bars(sol_res, "solar", "mae", fignum="4.4"),
        PL.learning_curve(dem_hist[h0], h0, fignum="4.5"),
        PL.feature_importance(sol_imp[ss0], ss0, fignum="4.6"),
        PL.daily_profile(dem_pred, h0, "demand", fignum="4.7"),
        PL.daily_profile(sol_pred, ss0, "solar", fignum="4.8"),
    ]
    for f in figs:
        print("   ", f.name)

    meta_out = {"data_mode": C.DATA_MODE, "n_homes": int(n_homes),
                "n_systems": int(n_systems), "seq_len": C.SEQ_LEN,
                "split": split, "runtime_seconds": round(time.time() - t0, 1)}
    (C.RESULTS / "run_metadata.json").write_text(json.dumps(meta_out, indent=2))
    banner(f"DONE in {meta_out['runtime_seconds']}s -> results/ figs/ models/")
    return dem_res, sol_res, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--homes", type=int, default=C.N_HOUSEHOLDS)
    ap.add_argument("--systems", type=int, default=C.N_PV_SYSTEMS)
    ap.add_argument("--pv-path", default=None,
                    help="folder containing uk_pv metadata.csv + 30_minutely/")
    ap.add_argument("--ideal-dir", default=None,
                    help="folder containing downloaded IDEAL csv/csv.gz files")
    a = ap.parse_args()

    ideal_files = None
    if a.pv_path or a.ideal_dir:
        if not (a.pv_path and a.ideal_dir):
            ap.error("--pv-path and --ideal-dir must be given together")
        C.DATA_MODE = "real"
        ideal_files = discover_ideal_files(a.ideal_dir)
        print(f"  discovered {len(ideal_files)} IDEAL files in {a.ideal_dir}")

    main(a.homes, a.systems, pv_path=a.pv_path, ideal_files=ideal_files)
