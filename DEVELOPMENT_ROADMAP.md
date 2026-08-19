# VoltSync AI — Development Roadmap

> **Read this file first.** It is the authoritative development plan for this
> repository. It is written to be actionable by both a human developer and an
> AI coding agent (Claude Code or equivalent).
>
> Companion file: `CLAUDE.md` (methodology rules that must never be broken).
> If this roadmap and `CLAUDE.md` ever disagree, `CLAUDE.md` wins.

---

## 0. Project identity

| Field | Value |
|---|---|
| **Title** | VoltSync AI: An AI Forecasting Engine for Residential Demand & Solar Generation |
| **Student** | Hamza Naveed · 35017370 |
| **Course** | MSc Data Science and AI |
| **Module** | Research Skills for Computing — 55-710248 |
| **Institution** | Sheffield Hallam University · Level 7 |
| **Supervisor** | Dr. Efosa Osagie |
| **Authoritative spec** | Chapters 1–3 of the approved proposal |
| **Ethics route** | UREC 1 — no human participants, secondary open data only |

### The research questions

| ID | Question | Model | Baseline |
|---|---|---|---|
| **RQ1** | How accurately can an LSTM forecast short-term residential electricity demand? | TensorFlow/Keras LSTM | Seasonal persistence |
| **RQ2** | How accurately can a gradient boosting model forecast short-term rooftop solar generation? | XGBoost | Seasonal persistence |

### The four objectives (§1.5)

| Obj | Requirement | Status |
|---|---|---|
| **i** | Review the literature | ✅ Done (Chapter 2) |
| **ii** | Obtain and prepare UK demand + rooftop solar datasets from public sources; clean, handle missing values, align to half-hourly | ⬜ **BLOCKING — code complete, never run on real data** |
| **iii** | Build and train the two models | ✅ Done (validated on synthetic) |
| **iv** | Evaluate against baselines using standard metrics, and discuss | ⚠️ Metrics done; discussion is Chapter 5 |

**Objective (ii) is the single blocking item. Everything in Phase 1 below exists to close it.**

---

## 1. SCOPE GUARDRAILS — read before writing any code

Chapter 1 §1.8 and Chapter 3 §3.8 **explicitly exclude** the following. Adding
any of them puts the project outside its approved scope and is a scope-control
failure, not extra credit.

### ❌ DO NOT BUILD

- **Any dashboard, web UI, or front end.** §3.8: *"Full deployment, such as a
  live dashboard or a community pilot, is outside the scope of this project and
  is noted as future work."*
- **Energy pricing** of any kind
- **Peer-to-peer trading** between households
- **Battery storage** design, sizing, or control
- **Optimisation / dispatch / scheduling** (no MILP, no RL, no linear programming)
- **Fairness metrics** (no Gini, no Lorenz curves)
- **Carbon intensity** modelling or carbon-aware anything
- **Any live, real-time, or deployed energy management system**

> **Note for AI agents:** an earlier iteration of this project *did* contain a
> Streamlit dashboard, a MILP optimiser, a P2P pricing engine and Gini fairness
> metrics. Those were **removed deliberately** because they contradict the
> approved proposal. If you find references to them in git history, old
> branches, or stale documentation, **do not restore them.** They belong only in
> the "future work" section of Chapter 5.

### ✅ IN SCOPE

Forecasting two quantities, evaluating them against a simple baseline, and
reporting the result. Nothing more.

---

## 2. Current state of the repository

### What exists and works

```
src/config.py       all tunables — change values HERE, never inline
src/data.py         download, clean, unit selection, synthetic generator
src/weather.py      Open-Meteo client — per-system, grid-deduplicated
src/features.py     calendar/lag features, chronological split, sequences
src/model_lstm.py   Keras LSTM + Scaler                        (§3.4.1)
src/model_gbm.py    XGBoost                                    (§3.4.2)
src/baselines.py    seasonal persistence                       (§3.4.3)
src/metrics.py      MAE / RMSE / MAPE / daylight-MAPE          (§3.7)
src/experiment.py   RQ1 + RQ2 runners, per-unit resume cache
src/plots.py        Chapter 4 figures
run_experiment.py   main entry point (CLI)
tests/              24 correctness tests
CLAUDE.md           methodology rules for AI agents
```

### Verify the baseline works before changing anything

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_pipeline.py                        # expect: ALL TESTS PASSED (24)
python run_experiment.py --homes 2 --systems 2       # synthetic smoke test
```

On Apple Silicon, substitute `tensorflow-macos` for `tensorflow`.
No GPU required — all 8 LSTMs train in roughly 15 minutes on CPU.

### Known-good synthetic baseline (for regression comparison only)

| Task | Model | MAE | RMSE | MAPE | Baseline MAE | Improvement |
|---|---|---|---|---|---|---|
| Demand (RQ1) | LSTM | 0.0285 | 0.0566 | 16.82% | 0.0462 | +38.9% |
| Solar (RQ2) | XGBoost | 0.0029 | 0.0059 | 3.74%¹ | 0.0114 | +74.6% |

¹ daytime-only. **These are synthetic numbers and are optimistic by
construction** — the generator derives irradiance from the same cloud process as
PV output, so the solar model has near-perfect information. Never quote them in
the dissertation. They exist only to detect code regressions.

---

## 3. METHODOLOGY RULES — violating these invalidates the results

Each rule maps to Chapter 3 and is covered by a test. **If a change breaks one
of these tests, the change is wrong — not the test.**

| # | Rule | Why | Test |
|---|---|---|---|
| 1 | **Never shuffle.** Splits are chronological. | A random split lets the model see the future (§3.5) | `test_splits_are_chronological` |
| 2 | **Fit scalers on the training split only**, then transform val/test | Fitting on the full series leaks future statistics backwards | `test_scaler_fitted_on_train_only` |
| 3 | **Solar MAPE is daylight-masked.** Use `daylight_mape`, never `mape`, on PV | Night generation is exactly zero → plain MAPE undefined (Table 3.2) | `test_mape_definitions` |
| 4 | **Lag features strictly backward-looking** | No feature may use the contemporaneous or future target | `test_no_target_leakage_in_features` |
| 5 | **All timestamps tz-aware UTC.** Convert to Europe/London only for display | BST/UTC mismatch silently shifts half the year by an hour | — |
| 6 | **`uk_pv` `datetime_GMT` is period-ending** (12:00 covers 11:30–12:00). Weather sampled at interval midpoint | Otherwise irradiance leads generation by half a step | `test_period_ending_shift` |
| 7 | **Every model judged against seasonal persistence** | A model is useful only where it clearly beats the baseline (§3.4.3) | `test_baseline_is_yesterday` |
| 8 | **`DATA_MODE="real"` must never fall back to synthetic** — it raises instead | Results must never be silently mislabelled | `test_real_mode_refuses_silent_fallback` |

**Run `python tests/test_pipeline.py` before and after every change.** Takes seconds.

---

## 4. Data sources

### 4.1 Primary sources (required)

| Source | Role | Coverage | Licence | Access |
|---|---|---|---|---|
| **openclimatefix/uk_pv** | Solar generation, **per system** | 2010–2025 | CC-BY-4.0, DOI 10.57967/hf/0878 | **Gated** — accept conditions on HF page first |
| **IDEAL Household Energy** | Residential demand | ~2016-09 → **2018-06** | CC BY, DOI 10.7488/ds/2836 | Edinburgh DataShare, DSpace REST API |
| **Open-Meteo** | Historical weather | 1940– | CC-BY 4.0 | Keyless, <10k calls/day |

### 4.2 Study windows — CRITICAL

**The two datasets do not overlap.** IDEAL ends June 2018; uk_pv runs to 2025.
The two tasks are independent (different units, different models, never joined),
so each gets its own window:

```python
# src/config.py
DEMAND_START, DEMAND_END = "2016-09-01", "2018-06-01"   # IDEAL
PV_START, PV_END         = "2022-01-01", "2024-01-01"   # uk_pv
```

> **Do not force these to a single window.** Doing so leaves **zero** usable
> IDEAL homes and `select_ideal_homes()` will raise. Guarded by
> `test_windows_match_dataset_coverage`.

### 4.3 Sheffield Solar PV_Live — supplementary only

The Sheffield Solar API at `https://www.solar.sheffield.ac.uk/api/` is **PV_Live**.
Researched and assessed:

- **It provides only AGGREGATED generation** — national, DNO Licence Area (14
  regions), and Grid Supply Point (~350 regions). **There is no per-system
  endpoint.**
- It therefore **cannot satisfy §3.3.2**, which requires individual rooftop
  systems with per-system kWp / tilt / orientation metadata.
- It **cannot replace uk_pv.** Keep uk_pv as the primary solar source.
- It **is** openly licensed (CC BY 4.0, no API key), so it is UREC 1 compatible
  and safe to use as a **supplementary validation / context reference**.

**Permitted uses (no scope change needed):**
- Sanity-check the seasonal and diurnal shape of your per-system data against
  the regional aggregate
- Contextualise a system's yield (kWh/kWp) against its region

**Forbidden without a scope amendment agreed with Dr. Osagie:**
- Making a PV_Live regional series a formal *forecasting target*
- Reporting PV_Live accuracy alongside per-system accuracy as if comparable —
  an aggregate series is far smoother and will post artificially strong metrics
  that do not answer RQ2

Technical notes if used: base URL `https://api.pvlive.uk/pvlive/api/v4/`,
national = `/gsp/0`, half-hourly, `generation_mw`, `datetime_gmt` is
**interval-ending UTC**, Python client `pip install pvlive-api`.
Cite: Huxley et al. (2022), *Renewable and Sustainable Energy Reviews*, 156, 112000.
Note PV_Live values are model estimates (±5.1% uncertainty) and are
retrospectively revised — log your download date.

**Sheffield Solar has no demand dataset relevant to this project.** The demand
side stays with IDEAL.

---

## 5. PHASE 1 — Get real data flowing (closes objective ii)

### T1 — Download uk_pv

**Prerequisite:** create a Hugging Face account, open the `uk_pv` dataset page,
**click to accept the conditions** (it is gated), then create an access token.

```python
from huggingface_hub import login
login(token="hf_xxxxxxxxxxxx")

from src.data import download_uk_pv
pv_path = download_uk_pv(years=(2022, 2023))
```

> ⚠️ **Never call `load_dataset("openclimatefix/uk_pv")`.** It raises
> `DatasetGenerationError`. The maintainers recommend reading the Parquet
> directly, which is what `download_uk_pv()` does.

Place the result at `data/raw/uk_pv/` containing `metadata.csv` and
`30_minutely/year=YYYY/`.

**Acceptance criteria**
- [ ] `data/raw/uk_pv/metadata.csv` exists and loads in pandas
- [ ] `30_minutely/year=2022/` and `year=2023/` contain `.parquet` files
- [ ] `select_pv_systems(metadata)` returns ≥ 8 rows

---

### T2 — Download IDEAL demand data

The Edinburgh DataShare web page uses bot detection; the REST API does not.

```bash
curl -s "https://datashare.ed.ac.uk/rest/handle/10283/3647/bitstreams"
```

```python
from src.data import download_ideal
download_ideal(handle="10283/3647",
               filename="home123_electric_combined.csv.gz", sequence=1)
```

> **Download 12–15 homes, not 8.** `select_ideal_homes()` filters on record
> completeness, so surplus candidates are required.

**Acceptance criteria**
- [ ] 12–15 IDEAL electricity files in `data/raw/ideal/`
- [ ] `discover_ideal_files("data/raw/ideal")` returns them keyed by home id

---

### T3 — Verify the IDEAL file format ⚠️ DO NOT SKIP

`load_ideal_home()` assumes a **two-column, header-less CSV** of
`timestamp, value`. That is what the IDEAL paper describes, but the actual file
layout is **unverified**. This is the single most likely thing to break.

```python
from src.data import peek_ideal_file
peek_ideal_file("data/raw/ideal/<one-file>.csv.gz")
```

| What you observe | Action in `load_ideal_home()` |
|---|---|
| 2 columns, timestamp + number, no header | None — proceed |
| A header row present | `header=None` → `header=0` |
| 3+ columns (e.g. sensor id) | Select the correct column by name/index |
| Unix epoch timestamps | Add `unit="s"` to `pd.to_datetime()` |
| Values in Watts vs Wh | Adjust the unit conversion; target is **kWh per half hour** |

**Only that one function should ever need changing.** Everything downstream
consumes a clean half-hourly kWh series.

**Acceptance criteria**
- [ ] Raw lines inspected and column layout known
- [ ] `load_ideal_home()` returns a plausible kWh series
- [ ] Typical half-hourly value in the range **0.03–0.5 kWh**
- [ ] Annual total per home in the range **1,800–4,500 kWh**

---

### T4 — First real run (small)

Start with two units each so failures surface fast.

```bash
python run_experiment.py --homes 2 --systems 2 \
    --pv-path data/raw/uk_pv --ideal-dir data/raw/ideal
```

Passing both paths sets `DATA_MODE="real"` automatically and auto-discovers the
IDEAL files.

**Troubleshooting**

| Error | Fix |
|---|---|
| `Only N homes meet 80% coverage` | Check `DEMAND_START`/`DEMAND_END` match IDEAL (~2016-09 → 2018-06), or lower `MIN_COVERAGE` |
| `Only N PV systems matched` | Widen `lat_pad`/`lon_pad` in `select_pv_systems()`, or relax the kWp band |
| Weather fetch failure | Open-Meteo rate limit; responses cache in `cache/`, wait and re-run |
| Demand parsing error | Return to T3 |

**Acceptance criteria**
- [ ] Run completes with `"data_mode": "real"` in `results/run_metadata.json`
- [ ] Solar cleaning log prints row counts per step
- [ ] No exceptions

---

### T5 — Sanity-check the real results ⚠️ MOST IMPORTANT REVIEW

Before scaling up, confirm the numbers are physically plausible.

| Check | Expected on real UK data |
|---|---|
| Solar specific yield | **750–950 kWh/kWp/year** |
| Household annual demand | **1,800–4,500 kWh** |
| Night-time solar generation | **exactly zero** |
| Demand peak hour | evening, ~17:00–19:00 |
| Solar daylight-MAPE | **15–30%** |
| Demand MAPE | **15–35%** per household |

> **Expect accuracy to fall versus the synthetic baseline. This is correct, not
> a failure.** Per §2.5 and §3.4.3, what matters is whether each model *clearly
> beats its baseline* — not the absolute MAPE. Low figures in the literature are
> typically for *aggregated* load; individual half-hourly household demand is
> far more volatile.

**Acceptance criteria**
- [ ] All physical checks pass
- [ ] Any unit failing to beat baseline is understood and explainable

---

### T6 — Full run and freeze outputs

```bash
python tests/test_pipeline.py
python run_experiment.py --homes 8 --systems 8 \
    --pv-path data/raw/uk_pv --ideal-dir data/raw/ideal

zip -r voltsync_final_results.zip results/ figs/ models/
git add -A && git commit -m "Final real-data run"
git tag results-final
```

Freezing matters: the numbers in the dissertation must not drift from the
numbers in the repository.

**Acceptance criteria**
- [ ] 8 homes and 8 systems complete on real data
- [ ] All 8 figures + 4 tables regenerated
- [ ] Results committed and tagged
- [ ] `results/RESULTS_SUMMARY.md` regenerated

---

### T7 (OPTIONAL) — PV_Live validation reference

Only after T6. Supplementary use only — see §4.3.

```bash
pip install pvlive-api
```

Suggested module: `src/pvlive_reference.py`

```python
"""
Supplementary validation reference (NOT a forecasting target).

PV_Live provides AGGREGATED GB solar generation only. It cannot satisfy §3.3.2
and must never be reported as comparable to per-system accuracy.

Sheffield Solar PV_Live, CC BY 4.0. Cite Huxley et al. (2022),
Renewable and Sustainable Energy Reviews, 156, 112000.
"""
from pvlive_api import PVLive

def regional_reference(start, end, gsp_id=0):
    """Half-hourly aggregated generation in MW. datetime_gmt is interval-ending UTC."""
    return PVLive().between(start=start, end=end, entity_type="gsp",
                            entity_id=gsp_id, dataframe=True)
```

**Acceptance criteria**
- [ ] Used only for shape validation / yield context
- [ ] **No PV_Live accuracy metric reported alongside per-system metrics**
- [ ] Download date logged; Huxley et al. (2022) added to references
- [ ] One sentence added to §3.3.2 or §3.3.3 naming PV_Live as a secondary reference

---

## 6. PHASE 2 — Write-up (after T6, per the agreed sequence)

### T8 — Chapter 4: Results
Presents, does not interpret.

| § | Section | Source |
|---|---|---|
| 4.1 | Introduction | — |
| 4.2 | Data preparation outcomes | cleaning logs, `table_3_1_pv_systems.csv` |
| 4.3 | Demand results (RQ1) | `table_4_1_demand_results.csv`, Figs 4.1, 4.3, 4.5, 4.7 |
| 4.4 | Solar results (RQ2) | `table_4_2_solar_results.csv`, Figs 4.2, 4.4, 4.6, 4.8 |
| 4.5 | Comparison against baselines | `table_4_3_summary.csv` |
| 4.6 | Summary | — |

State the daylight-mask explicitly where solar MAPE first appears, and say why.

### T9 — Chapter 5: Discussion and Conclusion
1. Why solar forecasts better than demand (measurable physics vs human behaviour)
2. Why MAPE above 5% is still a good result (aggregated vs individual load)
3. Practical implications (objective iv)
4. Limitations (observed not forecast weather §3.3.3; small sample; historical
   data) and future work — **this is where pricing, trading, battery control,
   optimisation and deployment belong**

### T10 — Rebuild the supporting documentation ⚠️ CORRECTNESS ISSUE
The current supporting document describes a **different project** — a
six-objective P2P energy management system with MILP dispatch and fairness
metrics. None of that appears in Chapters 1–3. Two submitted documents that
contradict each other is a real risk.

Must become: 4 objectives, forecasting only, correct title, supervisor
Dr. Efosa Osagie, course MSc Data Science and AI, UREC 1 ethics, literature
matrix, AI query log, risk register, Gantt, APA references.

---

## 7. PHASE 3 — Assessed deliverable (100% of module mark)

### T11 — Seven slides
1. Working title · 2. Introduction and justification · 3. RQ, aims, objectives,
deliverable · 4. Literature review findings (§2.5 gaps) · 5. Research design
(CRISP-DM, LSTM, GBM, baseline, metrics) · 6. Ethics, risks, issues (UREC 1) ·
7. Milestones · 8. References (APA)

Put **one real result** on slide 5 or 7.

### T12 — Record the presentation
- **12 minutes. Face must appear in the corner — a recording without it is an automatic fail.**
- Panopto. ~90 seconds per slide. Rehearse twice. Watch back in full.

### T13 — Pre-submission checklist
- [ ] Recording, 12 min, face visible
- [ ] Slides submitted (PowerPoint)
- [ ] Supporting doc matches Chapters 1–3 (T10)
- [ ] Signed UREC 1 form attached
- [ ] AITS transparency declaration as appendix
- [ ] References in full APA 7th
- [ ] Every number traces to a file in `results/`
- [ ] **No synthetic results quoted anywhere**
- [ ] Blackboard + Turnitin confirmed

---

## 8. Execution order

| Order | Tasks | Rationale |
|---|---|---|
| **1st** | T1–T6 | Nothing is citable until objective ii is closed |
| **2nd** | T10 | Correctness bug — cheap now, expensive later |
| **3rd** | T8–T9 | Needs frozen results from T6 |
| **4th** | T11–T13 | Carries the module mark |
| *Optional* | T7 | Only after T6 |

---

## 9. Instructions for AI coding agents

**Before any change**
1. Read `CLAUDE.md`
2. Run `python tests/test_pipeline.py` — establish a green baseline

**After any change**
1. Run `python tests/test_pipeline.py` again — all 24 must pass
2. If a test fails, **fix the code, not the test**, unless the test itself
   encodes a stale assumption (rare — justify it explicitly)

**Anti-patterns — do not do these**

| ❌ Don't | ✅ Do |
|---|---|
| Add `shuffle=True` anywhere | Keep chronological ordering |
| Call `scaler.fit()` on full data | Fit on train split only |
| Use `mape()` on solar | Use `daylight_mape()` |
| Remove the period-ending shift | Keep the 15-minute midpoint sampling |
| Build a dashboard / API / web UI | Out of scope (§3.8) |
| Add optimisation, pricing, batteries, fairness | Out of scope (§1.8) |
| Make PV_Live a forecasting target | Supplementary reference only |
| Hardcode values in modules | Put them in `src/config.py` |
| Quote synthetic results as findings | Re-run on real data first |
| Force one study window for both tasks | Keep `DEMAND_*` and `PV_*` separate |

**When uncertain about scope:** check §1.8 and §3.8. If a feature is not needed
to answer RQ1 or RQ2, it does not belong in this repository.

**When uncertain about data:** verify the file with `peek_ideal_file()` rather
than assuming the schema.

---

## 10. Quick reference

```bash
# tests
python tests/test_pipeline.py

# synthetic
python run_experiment.py --homes 8 --systems 8

# real
python run_experiment.py --homes 8 --systems 8 \
    --pv-path data/raw/uk_pv --ideal-dir data/raw/ideal

# inspect an IDEAL file before ingesting
python -c "from src.data import peek_ideal_file; peek_ideal_file('data/raw/ideal/x.csv.gz')"
```

**Outputs**
`results/table_4_1_demand_results.csv` · `results/table_4_2_solar_results.csv` ·
`results/table_4_3_summary.csv` · `results/RESULTS_SUMMARY.md` ·
`results/run_metadata.json` · `figs/fig4_*.png` · `models/`

**Citations**

- OpenClimateFix (2025). *UK PV dataset*. Hugging Face. DOI 10.57967/hf/0878. Data made possible by Sheffield Solar.
- Pullinger, M., et al. (2021). The IDEAL household energy dataset. *Scientific Data*, 8, 146. DOI 10.7488/ds/2836
- Huxley, O.T., et al. (2022). The uncertainties involved in measuring national solar photovoltaic electricity generation. *Renewable and Sustainable Energy Reviews*, 156, 112000. *(only if PV_Live is used)*
- Open-Meteo (2024). Weather API. CC-BY 4.0.
