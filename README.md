# VoltSync AI
### An AI Forecasting Engine for Residential Demand & Solar Generation

Hamza Naveed · 35017370 · MSc Data Science and AI
Research Skills for Computing (55-710248) · Sheffield Hallam University
Supervisor: Dr. Efosa Osagie

---

## Scope

Implements Chapters 1–3 exactly. Two short-term forecasting tasks, each
evaluated against a seasonal persistence baseline.

Per §1.8 this project deliberately **excludes** energy pricing, trading between
households, battery storage design or control, and any live or deployed energy
management system. Those are future work.

## Research questions

| | Question | Model | Answered in |
|---|---|---|---|
| **RQ1** | How accurately can an LSTM forecast short-term household demand vs a simple baseline? | LSTM (TensorFlow/Keras) | `results/table_4_1_demand_results.csv` |
| **RQ2** | How accurately can gradient boosting forecast rooftop solar vs a simple baseline? | XGBoost | `results/table_4_2_solar_results.csv` |

## Headline results

8 households, 8 PV systems, 2 years at 30-minute resolution, chronological
65/15/20 split.

| Task | Model | MAE | RMSE | MAPE | Baseline MAE | Improvement | Beat baseline |
|---|---|---|---|---|---|---|---|
| Demand (RQ1) | LSTM | 0.0285 | 0.0566 | 16.82% | 0.0462 | **+38.9%** | **8 / 8** |
| Solar (RQ2) | Gradient boosting | 0.0029 | 0.0059 | 3.74%¹ | 0.0114 | **+74.6%** | **8 / 8** |

¹ Solar MAPE is daytime-only (Table 3.2) — night generation is zero, so
percentage error is undefined there.

**Both models beat the baseline on every unit.** Solar is the more accurately
modelled task, which is the expected result: generation is driven by measurable
weather and solar geometry, whereas individual household demand is driven by
human behaviour and contains sharp, essentially unpredictable appliance events.

## Quick start

```bash
pip install -r requirements.txt
python tests/test_pipeline.py                          # 24 checks
python run_experiment.py --homes 8 --systems 8         # synthetic
```

### Running on real data locally

1. Download `uk_pv` (metadata.csv + `30_minutely/year=YYYY/`) into
   `data/raw/uk_pv/`
2. Download the IDEAL electricity files into `data/raw/ideal/`
3. Check the IDEAL layout before a full ingest:

```python
from src.data import peek_ideal_file
peek_ideal_file("data/raw/ideal/<one-file>.csv.gz")
```

4. Run:

```bash
python run_experiment.py --homes 8 --systems 8 \
    --pv-path data/raw/uk_pv --ideal-dir data/raw/ideal
```

Passing both paths sets `DATA_MODE="real"` automatically and auto-discovers the
IDEAL files. Training all 8 LSTMs takes roughly 15 minutes on CPU — no GPU
required.

Outputs land in `results/`, `figs/` and `models/`. Per-unit results are cached
in `results/partial/`, so an interrupted run resumes where it stopped — useful
on Colab, where sessions are capped at 12 hours.

## Switching to the real datasets

See `notebooks/VoltSync_AI_Real_Data_Run.ipynb` for the full sequence.

```python
import src.config as C
C.DATA_MODE = "real"
from src.data import load_data
pv, demand, meta = load_data(n_pv=8, n_homes=8,
                             pv_path=pv_path, ideal_files=ideal_files)
```

`load_data()` **refuses to fall back** to synthetic data in real mode — a
missing path raises rather than silently mislabelling results.

Real-data handling built in:

- **Per-system weather** (3.3.3) from each system's own coordinates, memoised
  per ~25 km reanalysis grid cell so co-located systems cost one request.
- **Period-ending correction** — `datetime_GMT` covers `[t-30min, t]`, so
  weather is sampled at the interval midpoint. Without it irradiance leads
  generation by half a step.
- **Unit selection** — `select_pv_systems()` and `select_ideal_homes()` pick
  units with long, mostly complete records (3.3.1, 3.3.2).
- **Cleaning logs** returned on `pv.attrs["cleaning_log"]` for Chapter 4.

```python
from src.data import download_uk_pv, select_pv_systems, load_uk_pv, clean_pv
path = download_uk_pv(years=(2022, 2023), token="hf_...")
```

**Do not call `load_dataset("openclimatefix/uk_pv")`** — it raises
`DatasetGenerationError`. Download the parquet directly. The dataset is gated:
accept the conditions on the dataset page first.

For demand, resolve the IDEAL bitstream list, then use `download_ideal()` and
`load_ideal_home()` (1-second data is resampled to 30 minutes first, which
removes the volume problem entirely).

## Method notes worth defending in the viva

- **Scalers are fitted on the training split only**, then applied. Fitting on
  the full series first would leak future statistics backwards.
- **Splits are chronological, never shuffled** (§3.5). Shuffling would let the
  model see the future during training.
- **Solar MAPE is daylight-masked.** Plain MAPE divides by the actual value and
  PV is exactly zero at night, so it is undefined.
- **Generation predictions are clipped at zero** — negative output is
  physically impossible.
- **Lag features are strictly backward-looking.**
- **All timestamps are tz-aware UTC.** The BST/UTC mismatch silently shifts half
  the year by an hour.

## Honest limitation of the bundled run

The bundled results use the **synthetic** data generator, which produces the
identical schema so every module is testable offline. Its solar irradiance
derives from the same cloud process as its PV output, so the solar model has
near-perfect information and its accuracy is **optimistic by construction**.
Expect materially higher error on genuine `uk_pv` data. The demand results are
more representative, because the synthetic demand contains genuinely
unpredictable stochastic appliance events.

## Tests

```bash
python tests/test_pipeline.py     # 21 checks
```

Covers leakage, split ordering, metric definitions, the period-ending shift,
weather grid de-duplication, and the real/synthetic mode boundary.

## Repository

```
src/config.py       all tunables
src/data.py         download, clean, synthetic generator
src/weather.py      Open-Meteo client (historical + forecast)
src/features.py     calendar/lag/weather features, chronological split
src/model_lstm.py   LSTM demand model (Keras)      §3.4.1
src/model_gbm.py    gradient boosting solar model  §3.4.2
src/baselines.py    seasonal persistence           §3.4.3
src/metrics.py      MAE / RMSE / MAPE              §3.7
src/experiment.py   RQ1 + RQ2 experiment runner
src/plots.py        Chapter 4 figures
src/experiment.py   RQ1 + RQ2 experiment runner
run_experiment.py   main entry point
tests/              correctness tests
```

## Study windows — important

The two tasks are **independent**: RQ1 uses IDEAL homes, RQ2 uses uk_pv
systems. They share the 30-minute time base but not a calendar window, and
nothing is ever joined across them. Separate windows are necessary because the
datasets do not overlap:

| Dataset | Coverage | Window used |
|---|---|---|
| IDEAL (demand) | ~2016-09 to **2018-06** | `DEMAND_START` / `DEMAND_END` |
| uk_pv (solar) | 2010 to 2025 | `PV_START` / `PV_END` |

Forcing a single window would leave **zero** usable IDEAL homes. A test
(`test_windows_match_dataset_coverage`) guards this.

## Data sources

| Source | Licence |
|---|---|
| openclimatefix/uk_pv (Sheffield Solar) | CC-BY-4.0, DOI 10.57967/hf/0878, gated |
| IDEAL household energy dataset | CC BY, DOI 10.7488/ds/2836 |
| Open-Meteo | CC-BY 4.0, keyless |
