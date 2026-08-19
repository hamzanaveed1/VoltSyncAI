"""
VoltSync AI - An AI Forecasting Engine for Residential Demand & Solar Generation
Central configuration.

Scope is deliberately bounded per Chapter 1 (1.8): two short-term forecasting
tasks evaluated against a persistence baseline. No pricing, no inter-household
trading, no battery control, no deployed system.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW, DATA_PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
MODELS, FIGS, RESULTS, CACHE = (ROOT / "models", ROOT / "figs",
                                ROOT / "results", ROOT / "cache")
for _p in (DATA_RAW, DATA_PROC, MODELS, FIGS, RESULTS, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data mode
#   "synthetic" -> generate series with the IDENTICAL schema (runs offline)
#   "real"      -> IDEAL demand + openclimatefix/uk_pv generation
DATA_MODE = "synthetic"

HF_REPO = "openclimatefix/uk_pv"
PV_RESOLUTION = "30_minutely"          # 30-min series is a cumulative meter sum
IDEAL_DOI = "10.7488/ds/2836"          # Pullinger et al. (2021)

# ---------------------------------------------------------------- study window
#
# The two forecasting tasks are INDEPENDENT: RQ1 uses IDEAL homes, RQ2 uses
# uk_pv systems. They share the 30-minute time base (3.5) but not a calendar
# window, and nothing is ever joined across them. Separate windows are therefore
# required, because the datasets simply do not overlap:
#
#   IDEAL  : 23 months ending June 2018   (Pullinger et al., 2021)
#   uk_pv  : 2010 - 2025
#
# Forcing a single window would leave zero usable IDEAL homes.
DEMAND_START, DEMAND_END = "2016-09-01", "2018-06-01"     # IDEAL
PV_START, PV_END = "2022-01-01", "2025-04-01"             # uk_pv

# Backwards-compatible aliases (synthetic mode uses one window for both)
START, END = PV_START, PV_END
FREQ_MIN = 30
STEPS_PER_DAY = 24 * 60 // FREQ_MIN    # 48

# ---------------------------------------------------------------- sample sizes
# 3.3.1 / 3.3.2: "a small sample ... with long and mostly complete records"
N_HOUSEHOLDS = 8                       # IDEAL homes (demand task)
N_PV_SYSTEMS = 8                       # uk_pv systems (solar task)
KWP_MIN, KWP_MAX = 2.0, 4.0
REGION_LAT, REGION_LON = 53.38, -1.47  # Sheffield

# ---------------------------------------------------------------- forecasting
FORECAST_HORIZON = 1                   # next half hour (1.9: forecast horizon)
SEQ_LEN = 48                           # 24 h lookback window for the LSTM

# LSTM (3.4.1) - TensorFlow / Keras
LSTM_UNITS = 48
LSTM_UNITS_2 = 24
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 25
LSTM_BATCH = 384
LSTM_LR = 1e-3
LSTM_PATIENCE = 5

# Gradient boosting (3.4.2) - XGBoost
XGB_PARAMS = dict(
    n_estimators=1500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
    early_stopping_rounds=50, eval_metric="rmse", random_state=42,
)

# Chronological split (3.5) - never shuffled
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.65, 0.15, 0.20

# Metrics (3.7). Solar MAPE is daytime-only; night values are zero.
DAYLIGHT_MIN_KWH = 0.05

# uk_pv datetime_GMT is PERIOD-ENDING: a 12:00 row covers 11:30-12:00.
# Weather must therefore be sampled at the interval MIDPOINT, or irradiance
# leads generation by half a step and silently degrades the solar model.
PV_PERIOD_ENDING = True

# ERA5 grid is ~0.25 deg (~25 km). Rounding system coordinates to that grid
# de-duplicates weather requests across nearby systems, which keeps the run
# inside the Open-Meteo free-tier rate limit.
WEATHER_GRID_DEG = 0.25

# Minimum fraction of the study window a unit must cover to be selected
# (3.3.1 / 3.3.2: "long and mostly complete records").
MIN_COVERAGE = 0.80

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
API_TIMEOUT_S = 20
SEED = 42
