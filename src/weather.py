"""
Weather data (Chapter 3.3.3).

Historical weather is retrieved for the approximate location of EACH selected
solar system and matched to the generation records in time. Open-Meteo is used:
keyless, CC-BY 4.0, and the historical archive and live forecast endpoints share
an identical JSON schema.

Two correctness details handled here:

1. PERIOD-ENDING TIMESTAMPS. uk_pv's datetime_GMT is period-ending, so a 12:00
   row covers 11:30-12:00. Weather is therefore sampled at the interval
   MIDPOINT (t - 15 min). Without this, irradiance leads generation by half a
   step and quietly degrades the solar model.

2. GRID DE-DUPLICATION. ERA5 resolution is ~0.25 deg (~25 km), so nearby
   systems share a grid cell. Coordinates are rounded to that grid before
   requesting, which collapses redundant calls and keeps the run inside the
   free-tier rate limit.

Note (stated as a limitation in 3.3.3): observed weather is used, not forecast
weather. A deployed system would depend on forecasts.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C

HOURLY_VARS = ["temperature_2m", "shortwave_radiation", "cloud_cover",
               "wind_speed_10m", "relative_humidity_2m"]

_MEM: dict = {}          # in-process cache, keyed by grid cell


# ---------------------------------------------------------------- helpers
def grid_key(lat: float, lon: float, step=C.WEATHER_GRID_DEG):
    """Round to the reanalysis grid so nearby systems share one request."""
    return (round(round(lat / step) * step, 4), round(round(lon / step) * step, 4))


def _cache_path(lat, lon, start=None, end=None):
    start = (start or C.PV_START)[:10]
    end = (end or C.PV_END)[:10]
    return C.CACHE / f"weather_{lat}_{lon}_{start}_{end}.parquet"


def _to_frame(js: dict) -> pd.DataFrame:
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").astype(float)


# ---------------------------------------------------------------- fetching
def fetch_archive(lat, lon, start=None, end=None) -> pd.DataFrame:
    """Historical ERA5 reanalysis for one location, cached to disk."""
    start = start or C.PV_START
    end = end or C.PV_END
    lat, lon = grid_key(lat, lon)
    cp = _cache_path(lat, lon, start, end)
    if cp.exists():
        return pd.read_parquet(cp)
    import requests
    params = dict(latitude=lat, longitude=lon, start_date=start[:10],
                  end_date=end[:10], hourly=",".join(HOURLY_VARS),
                  timezone="UTC")
    r = requests.get(C.OPEN_METEO_ARCHIVE, params=params, timeout=C.API_TIMEOUT_S)
    r.raise_for_status()
    df = _to_frame(r.json())
    df.to_parquet(cp)
    return df


def fetch_forecast(lat, lon, days=2) -> pd.DataFrame:
    """Live forecast. Same schema as fetch_archive."""
    import requests
    params = dict(latitude=lat, longitude=lon, hourly=",".join(HOURLY_VARS),
                  forecast_days=days, timezone="UTC")
    r = requests.get(C.OPEN_METEO_FORECAST, params=params, timeout=C.API_TIMEOUT_S)
    r.raise_for_status()
    return _to_frame(r.json())


# ---------------------------------------------------------------- alignment
def align_to_index(hourly: pd.DataFrame, idx: pd.DatetimeIndex,
                   period_ending=C.PV_PERIOD_ENDING) -> pd.DataFrame:
    """
    Interpolate hourly weather onto the half-hourly target index.

    With period-ending stamps the value at t describes [t - 30min, t], so the
    representative weather is at the midpoint t - 15 min.
    """
    sample_at = idx - pd.Timedelta(minutes=C.FREQ_MIN / 2) if period_ending else idx
    union = hourly.index.union(sample_at).sort_values()
    out = hourly.reindex(union).interpolate("time").reindex(sample_at)
    out.index = idx            # relabel back onto the PV/demand index
    return out


# ---------------------------------------------------------------- synthetic
def synth_weather(idx: pd.DatetimeIndex, lat=C.REGION_LAT, lon=C.REGION_LON,
                  seed=C.SEED) -> pd.DataFrame:
    """Offline stand-in with the same columns as the API response."""
    from .data import _solar_position, _cloud
    rng = np.random.default_rng(seed + int(abs(lat * 1000)) % 997)
    cz = _solar_position(idx, lat, lon)
    k = _cloud(len(idx), rng)
    swr = 1000.0 * cz ** 1.18 * k
    doy = idx.dayofyear.values
    hr = idx.hour.values + idx.minute.values / 60.0
    temp = (10.5 - 7.0 * np.cos(2 * np.pi * (doy - 20) / 365.25)
            - 3.2 * np.cos(2 * np.pi * (hr - 15) / 24.0)
            + rng.normal(0, 1.1, len(idx)))
    return pd.DataFrame({
        "temperature_2m": np.round(temp, 2),
        "shortwave_radiation": np.round(swr, 1),
        "cloud_cover": np.round(np.clip((1 - k) * 135, 0, 100), 1),
        "wind_speed_10m": np.round(np.clip(rng.gamma(2.2, 2.4, len(idx)), 0, 30), 2),
        "relative_humidity_2m": np.round(
            np.clip(76 + rng.normal(0, 9, len(idx)) - 0.02 * swr, 25, 100), 1),
    }, index=idx)


# ---------------------------------------------------------------- entry point
def get_weather_at(lat: float, lon: float, idx: pd.DatetimeIndex,
                   period_ending=C.PV_PERIOD_ENDING) -> pd.DataFrame:
    """
    Weather for ONE location on the target index. Memoised per grid cell, so
    calling this once per PV system costs one request per distinct cell.
    """
    key = (*grid_key(lat, lon), len(idx), idx[0], idx[-1], period_ending)
    if key in _MEM:
        return _MEM[key]
    if C.DATA_MODE == "real":
        try:
            hourly = fetch_archive(lat, lon,
                                   start=str(idx[0].date()),
                                   end=str((idx[-1] + pd.Timedelta(days=1)).date()))
            out = align_to_index(hourly, idx, period_ending)
        except Exception as e:
            raise RuntimeError(
                f"Weather fetch failed for ({lat}, {lon}): {e}. Refusing to "
                "substitute synthetic weather while DATA_MODE='real'.") from e
    else:
        out = synth_weather(idx, lat, lon)
    _MEM[key] = out
    return out


def get_weather(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Region-level weather, used for the demand task."""
    return get_weather_at(C.REGION_LAT, C.REGION_LON, idx)
