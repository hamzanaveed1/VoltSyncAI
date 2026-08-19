# VoltSync AI — project context for Claude Code

MSc research project (SHU 55-710248). Two short-term forecasting tasks,
evaluated against a baseline. Implements Chapters 1–3 of the dissertation.

## Run it

```bash
python tests/test_pipeline.py                 # 24 checks — run before and after changes
python run_experiment.py --homes 8 --systems 8            # synthetic
python run_experiment.py --homes 8 --systems 8 \
    --pv-path data/raw/uk_pv --ideal-dir data/raw/ideal    # real data
```

## Scope — do NOT add these

Chapter 1 §1.8 explicitly excludes: energy pricing, peer-to-peer trading
between households, battery storage design or control, and any live or
deployed energy management system. Adding them puts the project outside its
approved scope. They are future work.

## Methodology rules that must not be broken

These come from Chapter 3 and are each covered by a test. If a change breaks
one, the change is wrong — not the test.

1. **Never shuffle the data.** Splits are chronological (§3.5). A random split
   lets the model see the future.
2. **Fit scalers on the training split only**, then transform val/test.
   Fitting on the full series leaks future statistics backwards.
3. **Solar MAPE is daylight-masked** (Table 3.2). Night generation is exactly
   zero, so plain MAPE is undefined. Use `daylight_mape`, never `mape`, on PV.
4. **Lag features are strictly backward-looking.** No feature may use the
   contemporaneous or future target.
5. **All timestamps are tz-aware UTC.** Convert to Europe/London only for
   display. The BST/UTC mismatch silently shifts half the year by an hour.
6. **`uk_pv` `datetime_GMT` is period-ending** — a 12:00 row covers
   11:30–12:00. Weather is sampled at the interval midpoint. Do not remove
   this shift.
7. **Every model is judged against seasonal persistence** (§3.4.3). A model is
   reported as useful only where it clearly beats that baseline.
8. **`DATA_MODE="real"` must never fall back to synthetic.** It raises instead,
   so results can never be silently mislabelled.

## Study windows

The two tasks are independent and the datasets do not overlap:

| Dataset | Coverage | Config |
|---|---|---|
| IDEAL (demand) | ~2016-09 → **2018-06** | `DEMAND_START/END` |
| uk_pv (solar) | 2010 → 2025 | `PV_START/END` |

Do not force them to a single window — that leaves zero usable IDEAL homes.

## Models

- Demand → **TensorFlow/Keras LSTM** (§3.4.1). Not PyTorch.
- Solar → **XGBoost** (§3.4.2).
- Baseline → seasonal persistence, value at the same time yesterday.

## Layout

```
src/config.py       all tunables — change here, not inline
src/data.py         download, clean, select units, synthetic generator
src/weather.py      Open-Meteo, per-system, grid-deduplicated
src/features.py     calendar/lag features, chronological split, sequences
src/model_lstm.py   LSTM + Scaler
src/model_gbm.py    XGBoost
src/baselines.py    seasonal persistence
src/metrics.py      MAE / RMSE / MAPE / daylight-MAPE
src/experiment.py   RQ1 + RQ2 runners, per-unit resume cache
src/plots.py        Chapter 4 figures
tests/              correctness tests
```

## First thing to check on real data

`load_ideal_home()` assumes a two-column header-less CSV (timestamp, value).
Verify before a full ingest:

```python
from src.data import peek_ideal_file
peek_ideal_file("data/raw/ideal/<file>.csv.gz")
```

If the layout differs, that one function is the only thing to change.
