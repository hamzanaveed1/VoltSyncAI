"""
Data acquisition and preprocessing (Chapter 3.3 and 3.5).

No new data is collected from users. All data is secondary, from open datasets,
downloaded within their licence terms (3.5, 3.6).

Preprocessing follows 3.5 in order:
  1. clean       - flag missing; interpolate short gaps, drop long gaps; remove
                   negative demand/generation and implausibly high generation;
                   force night-time solar to zero
  2. align       - common half-hourly step, single timezone (UTC)
  3. features    - built in features.py
  4. split       - chronological, never shuffled
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C

RNG = np.random.default_rng(C.SEED)
MAX_GAP_STEPS = 4          # interpolate gaps up to 2 h; drop anything longer


# ===========================================================================
# REAL DATA
# ===========================================================================
def download_uk_pv(years=(2022, 2023), token: str | None = None) -> str:
    """
    Download only the partitions needed. Do NOT use
    load_dataset("openclimatefix/uk_pv") - it raises DatasetGenerationError.
    The maintainers recommend reading the parquet directly. Dataset is gated:
    accept the conditions on the dataset page first.
    """
    from huggingface_hub import snapshot_download, login
    if token:
        login(token=token)
    patterns = ["metadata.csv", "bad_data.csv"]
    for y in years:
        patterns.append(f"{C.PV_RESOLUTION}/year={y}/*")
    return snapshot_download(repo_id=C.HF_REPO, repo_type="dataset",
                             allow_patterns=patterns,
                             local_dir=str(C.DATA_RAW / "uk_pv"))


def select_pv_systems(metadata: pd.DataFrame, n=C.N_PV_SYSTEMS,
                      lat_pad=0.4, lon_pad=0.6) -> pd.DataFrame:
    """Small sample of domestic systems with long, mostly complete records (3.3.2)."""
    m = metadata.copy()
    m["start_datetime_GMT"] = pd.to_datetime(m["start_datetime_GMT"], utc=True, errors="coerce")
    m["end_datetime_GMT"] = pd.to_datetime(m["end_datetime_GMT"], utc=True, errors="coerce")
    sel = m[
        m["kWp"].between(C.KWP_MIN, C.KWP_MAX)
        & m["latitude_rounded"].between(C.REGION_LAT - lat_pad, C.REGION_LAT + lat_pad)
        & m["longitude_rounded"].between(C.REGION_LON - lon_pad, C.REGION_LON + lon_pad)
        & (m["start_datetime_GMT"] <= pd.Timestamp(C.PV_START, tz="UTC"))
        & (m["end_datetime_GMT"] >= pd.Timestamp(C.PV_END, tz="UTC"))
    ]
    return sel.head(n)


def load_uk_pv(path: str, ss_ids: list[int]) -> pd.DataFrame:
    """
    Lazy scan with Hive partition pruning AND a timestamp filter.

    datetime_GMT is stored as tz-aware UTC with nanosecond precision. The
    filter bounds must be cast to match exactly - polars will not implicitly
    widen a naive/microsecond literal to compare against it.
    """
    import polars as pl
    lf = pl.scan_parquet(f"{path}/{C.PV_RESOLUTION}/**/*.parquet",
                         hive_partitioning=True,
                         hive_schema={"year": pl.Int16, "month": pl.Int8})
    dtype = pl.Datetime("ns", "UTC")
    start = pl.lit(pd.Timestamp(C.PV_START, tz="UTC")).cast(dtype)
    end = pl.lit(pd.Timestamp(C.PV_END, tz="UTC")).cast(dtype)
    return (lf.filter(
                (pl.col("datetime_GMT") >= start)
                & (pl.col("datetime_GMT") < end)
                & (pl.col("ss_id").is_in(ss_ids)))
            .collect().to_pandas())


def clean_pv(df: pd.DataFrame, metadata: pd.DataFrame,
             bad_data: pd.DataFrame | None = None):
    """
    Cleaning per 3.5 and the dataset card. Returns (clean_df, log).
    The log becomes a table in Chapter 4.
    """
    log, n0 = [], len(df)
    df = df.copy()
    df["datetime_GMT"] = pd.to_datetime(df["datetime_GMT"], utc=True)

    if bad_data is not None and len(bad_data):
        before = len(df)
        for _, r in bad_data.iterrows():
            df = df[~((df["ss_id"] == r["ss_id"])
                      & (df["datetime_GMT"] >= pd.Timestamp(r["start_datetime_GMT"], tz="UTC"))
                      & (df["datetime_GMT"] <= pd.Timestamp(r["end_datetime_GMT"], tz="UTC")))]
        log.append(("bad_data.csv windows removed", before - len(df)))

    before = len(df); df = df[df["generation_Wh"] >= 0]
    log.append(("negative generation removed", before - len(df)))

    before = len(df)
    cap = metadata.set_index("ss_id")["kWp"]
    df = df[df["generation_Wh"] <= df["ss_id"].map(cap) * 750.0]
    log.append(("implausible generation removed (> kWp x 750)", before - len(df)))

    # Night-time generation forced to zero (3.5)
    hour = df["datetime_GMT"].dt.hour
    night = (hour <= 2) | (hour >= 23)
    n_night = int((night & (df["generation_Wh"] > 0)).sum())
    df.loc[night, "generation_Wh"] = 0.0
    log.append(("night-time readings forced to zero", n_night))

    log.append(("rows retained", len(df)))
    log.append(("rows removed", n0 - len(df)))
    return df, log


def download_ideal(handle="10283/3647", filename=None, sequence=1) -> str:
    """
    IDEAL household energy dataset (Pullinger et al. 2021, DOI 10.7488/ds/2836).
    Resolve the exact filename first:
      curl https://datashare.ed.ac.uk/rest/handle/10283/3647/bitstreams
    isAllowed=y bypasses the click-through licence gate.
    """
    import requests
    url = (f"https://datashare.ed.ac.uk/bitstream/handle/{handle}/{filename}"
           f"?sequence={sequence}&isAllowed=y")
    dest = C.DATA_RAW / filename
    with requests.get(url, stream=True, timeout=300, verify=False) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return str(dest)


def load_ideal_home(path: str, home_id: int | str, chunksize: int = 2_000_000) -> pd.Series:
    """
    IDEAL electricity files are gzipped 2-column CSVs (timestamp UTC, value),
    no header, at 1-second resolution - several million rows and a common
    cause of out-of-memory crashes if read in one pass on modest hardware.
    Read and resample in chunks so peak memory is bounded by chunksize, not
    file size; sums and counts combine correctly across chunk boundaries
    because resample bins are re-aggregated by index after concatenation.
    """
    sums, counts = [], []
    reader = pd.read_csv(path, header=None, names=["timestamp", "value"],
                         compression="infer", chunksize=chunksize)
    for chunk in reader:
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True)
        binned = chunk.set_index("timestamp")["value"].astype(float) \
                      .resample(f"{C.FREQ_MIN}min")
        sums.append(binned.sum())
        counts.append(binned.count())
    sum_s = pd.concat(sums).groupby(level=0).sum()
    cnt_s = pd.concat(counts).groupby(level=0).sum()
    # 1-second Watts -> kWh per half hour
    kwh = (sum_s / cnt_s).sort_index() * (C.FREQ_MIN / 60.0) / 1000.0
    kwh.name = f"home_{home_id}"
    return kwh


def clean_demand(s: pd.Series):
    """Clean a demand series per 3.5. Returns (series, log)."""
    log, n0 = [], len(s)
    before = int(s.isna().sum())
    s = s[~s.index.duplicated(keep="first")].sort_index()
    s = s.asfreq(f"{C.FREQ_MIN}min")
    log.append(("missing values flagged", int(s.isna().sum())))

    neg = int((s < 0).sum())
    s = s.mask(s < 0)
    log.append(("negative demand removed", neg))

    # Interpolate short gaps only; long gaps stay NaN and are dropped downstream
    s = s.interpolate("time", limit=MAX_GAP_STEPS, limit_area="inside")
    log.append(("short gaps interpolated (<= 2 h)", int(before - s.isna().sum())))
    log.append(("long gaps left as missing", int(s.isna().sum())))
    log.append(("rows retained", int(s.notna().sum())))
    return s, log


# ===========================================================================
# SYNTHETIC DATA - identical schema, so every module is testable offline
# ===========================================================================
def _solar_position(idx, lat, lon):
    doy = idx.dayofyear.values.astype(float)
    fh = idx.hour.values + idx.minute.values / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (fh - 12) / 24.0)
    eq = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                   - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
           - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
           - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = fh * 60.0 + eq + 4.0 * lon
    ha = np.radians(tst / 4.0 - 180.0)
    latr = np.radians(lat)
    cz = np.sin(latr) * np.sin(dec) + np.cos(latr) * np.cos(dec) * np.cos(ha)
    return np.clip(cz, 0.0, None)


def _cloud(n, rng, phi=0.985):
    x = np.zeros(n); x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, np.sqrt(1 - phi ** 2))
    return np.clip(0.55 + 0.42 / (1 + np.exp(-1.5 * x)), 0.05, 1.0)


def generate_pv(idx, meta, rng):
    cz = _solar_position(idx, meta["latitude_rounded"], meta["longitude_rounded"])
    tilt = np.radians(meta["tilt"]); az = np.radians(meta["orientation"] - 180.0)
    gain = (np.cos(tilt) + np.sin(tilt) * 0.55) * (1 - 0.18 * abs(np.sin(az)))
    poa = 1000.0 * cz ** 1.18 * _cloud(len(idx), rng) * gain
    derate = 1.0 - 0.004 * np.clip(poa / 1000.0 * 25.0 - 5.0, 0, None)
    kw = np.clip(meta["kWp"] * poa / 1000.0 * 0.73 * derate, 0, meta["kWp"] * 0.92)
    kw[cz <= 0.001] = 0.0
    return np.round(kw * 1000.0 * (C.FREQ_MIN / 60.0), 1)     # Wh


def generate_demand(idx, annual_kwh, rng):
    h = idx.hour.values + idx.minute.values / 60.0
    dow, doy = idx.dayofweek.values, idx.dayofyear.values
    base = 0.30 + 0.85 * np.exp(-0.5 * ((h - 7.5) / 1.15) ** 2) \
                + 1.35 * np.exp(-0.5 * ((h - 18.5) / 2.05) ** 2)
    wknd = 0.42 + 0.55 * np.exp(-0.5 * ((h - 10.0) / 2.4) ** 2) \
                + 1.25 * np.exp(-0.5 * ((h - 18.8) / 2.6) ** 2)
    prof = np.where(dow >= 5, wknd, base)
    prof *= 1.0 + 0.30 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    spikes = np.zeros(len(idx))
    pos = rng.integers(0, len(idx), int(len(idx) * 0.045))
    spikes[pos] = rng.gamma(2.0, 0.45, len(pos))
    prof = np.clip((prof + spikes) * np.exp(rng.normal(0, 0.10, len(idx))), 0.04, None)
    kwh = prof * (C.FREQ_MIN / 60.0)
    yrs = len(idx) / (365.25 * C.STEPS_PER_DAY)
    return np.round(kwh * annual_kwh / (kwh.sum() / yrs), 5)


def build_synthetic(n_pv=C.N_PV_SYSTEMS, n_homes=C.N_HOUSEHOLDS):
    """Returns (pv_df, demand_df, pv_metadata) in the real schemas."""
    rng = np.random.default_rng(C.SEED)
    idx = pd.date_range(C.START, C.END, freq=f"{C.FREQ_MIN}min", tz="UTC",
                        inclusive="left")
    meta = pd.DataFrame({
        "ss_id": np.arange(10001, 10001 + n_pv),
        "start_datetime_GMT": pd.Timestamp(C.START, tz="UTC"),
        "end_datetime_GMT": pd.Timestamp(C.END, tz="UTC"),
        "latitude_rounded": np.round(C.REGION_LAT + rng.normal(0, .12, n_pv), 3),
        "longitude_rounded": np.round(C.REGION_LON + rng.normal(0, .18, n_pv), 3),
        "orientation": np.round(rng.uniform(140, 225, n_pv), 1),
        "tilt": np.round(rng.uniform(25, 42, n_pv), 1),
        "kWp": np.round(rng.uniform(C.KWP_MIN, C.KWP_MAX, n_pv), 2),
    })
    pv = pd.concat([pd.DataFrame({"ss_id": r["ss_id"], "datetime_GMT": idx,
                                  "generation_Wh": generate_pv(idx, r, rng)})
                    for _, r in meta.iterrows()], ignore_index=True)
    annuals = rng.uniform(2100, 4200, n_homes)
    dem = pd.DataFrame({f"home_{i+1}": generate_demand(idx, a, rng)
                        for i, a in enumerate(annuals)}, index=idx)
    dem.index.name = "time"
    # Demand annuals belong to homes, not PV systems - keep them separate so
    # the two sample sizes can differ independently.
    dem.attrs["annual_kwh"] = np.round(annuals, 0).tolist()
    return pv, dem, meta


# ===========================================================================
# UNIT SELECTION (3.3.1) and REAL-DATA ASSEMBLY
# ===========================================================================
def select_ideal_homes(home_series: dict, n=C.N_HOUSEHOLDS,
                       min_coverage=C.MIN_COVERAGE) -> list:
    """
    Choose IDEAL dwellings with long, mostly complete records (3.3.1).

    home_series : {home_id: pd.Series} of half-hourly demand
    Ranks by coverage of the study window and returns the best `n` ids.
    """
    full = pd.date_range(C.DEMAND_START, C.DEMAND_END, freq=f"{C.FREQ_MIN}min",
                         tz="UTC", inclusive="left")
    scored = []
    for hid, s in home_series.items():
        s = s[(s.index >= full[0]) & (s.index <= full[-1])]
        cov = float(s.notna().sum()) / len(full)
        if cov >= min_coverage:
            scored.append((hid, cov))
    scored.sort(key=lambda x: -x[1])
    if len(scored) < n:
        raise ValueError(
            f"Only {len(scored)} homes meet {min_coverage:.0%} coverage of "
            f"{C.DEMAND_START}..{C.DEMAND_END}; need {n}. IDEAL covers roughly "
            f"2016-09 to 2018-06 (Pullinger et al., 2021) - check "
            f"C.DEMAND_START / C.DEMAND_END, or lower C.MIN_COVERAGE.")
    return [hid for hid, _ in scored[:n]]


def build_real(pv_path: str, ideal_files: dict, n_pv=C.N_PV_SYSTEMS,
               n_homes=C.N_HOUSEHOLDS, verbose=True):
    """
    Assemble the real datasets into the SAME (pv, demand, meta) triple that
    build_synthetic() returns, so nothing downstream changes.

    pv_path     : path returned by download_uk_pv()
    ideal_files : {home_id: path_to_gzipped_csv}
    """
    meta_all = pd.read_csv(f"{pv_path}/metadata.csv")
    try:
        bad = pd.read_csv(f"{pv_path}/bad_data.csv")
    except Exception:
        bad = None

    meta = select_pv_systems(meta_all, n=n_pv)
    if len(meta) < n_pv:
        raise ValueError(f"Only {len(meta)} PV systems matched the selection "
                         f"criteria; need {n_pv}. Widen lat/lon padding.")
    if verbose:
        print(f"  selected {len(meta)} PV systems "
              f"({meta.kWp.min():.2f}-{meta.kWp.max():.2f} kWp)")

    pv = load_uk_pv(pv_path, meta.ss_id.tolist())
    pv, pv_log = clean_pv(pv, meta, bad)
    if verbose:
        for step, k in pv_log:
            print(f"    {step:45} {k:>10,}")

    # Demand: load every candidate home, then select on coverage
    series, logs = {}, {}
    for hid, path in ideal_files.items():
        s = load_ideal_home(path, hid)
        s, lg = clean_demand(s)
        series[hid], logs[hid] = s, lg
    chosen = select_ideal_homes(series, n=n_homes)
    if verbose:
        print(f"  selected {len(chosen)} IDEAL homes: {chosen}")

    idx = pd.date_range(C.DEMAND_START, C.DEMAND_END, freq=f"{C.FREQ_MIN}min",
                        tz="UTC", inclusive="left")
    demand = pd.DataFrame({f"home_{h}": series[h].reindex(idx)
                           for h in chosen}, index=idx)
    demand.index.name = "time"
    demand.attrs["cleaning_log"] = {h: logs[h] for h in chosen}
    pv.attrs["cleaning_log"] = pv_log
    return pv, demand, meta.reset_index(drop=True)


def load_data(n_pv=C.N_PV_SYSTEMS, n_homes=C.N_HOUSEHOLDS,
              pv_path=None, ideal_files=None, verbose=True):
    """
    Single entry point honouring C.DATA_MODE. This is the switch the rest of
    the pipeline goes through, so "real" can never silently fall back to
    synthetic.
    """
    if C.DATA_MODE == "real":
        if not pv_path or not ideal_files:
            raise ValueError(
                "DATA_MODE='real' requires pv_path (from download_uk_pv) and "
                "ideal_files={home_id: path}. Refusing to fall back to "
                "synthetic data, which would silently mislabel the results.")
        return build_real(pv_path, ideal_files, n_pv, n_homes, verbose)
    if verbose:
        print("  DATA_MODE='synthetic' - results are NOT from real UK data")
    return build_synthetic(n_pv=n_pv, n_homes=n_homes)


def discover_ideal_files(dirpath, pattern="*.csv*") -> dict:
    """
    Scan a folder of downloaded IDEAL files and build the {home_id: path} map
    that build_real() expects.

    Handles the usual IDEAL naming (home123_electric_combined.csv.gz) and falls
    back to the first integer found in the filename.
    """
    import re
    from pathlib import Path
    out = {}
    for p in sorted(Path(dirpath).glob(pattern)):
        m = re.search(r"home[_-]?(\d+)", p.name, re.I) or re.search(r"(\d+)", p.name)
        if m:
            out[int(m.group(1))] = str(p)
    if not out:
        raise FileNotFoundError(
            f"No IDEAL files matched {pattern} in {dirpath}. Expected gzipped "
            f"CSVs such as home123_electric_combined.csv.gz")
    return out


def peek_ideal_file(path, n=3):
    """
    Print the first few raw lines of an IDEAL file.

    ALWAYS run this before a full ingest. load_ideal_home() assumes a
    two-column, header-less CSV (timestamp, value). If the real layout differs,
    that one function is the only thing that needs changing.
    """
    import gzip, io
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", errors="replace") as f:
        lines = [next(f).rstrip("\n") for _ in range(n)]
    print(f"--- {path} ---")
    for i, l in enumerate(lines):
        print(f"  line {i}: {l[:120]}")
    print(f"  columns detected: {len(lines[0].split(','))}")
    return lines
