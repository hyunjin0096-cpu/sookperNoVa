"""Leakage-safe time features for any split. NO use of future rows / labels.

For each sample t, uses ONLY experiment's observations at times <= t.
The numerical time windows use actual seconds, never last N rows.
The input order is arbitrary; result aligns with input rows (sample_id).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

DEFAULT_SENSORS = [
    "meas_v", "meas_t1", "meas_t2", "meas_p", "meas_ch4", "meas_co", "meas_co2",
    "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o", "meas_no", "meas_no2",
    "meas_so2", "atmp_t", "thermal_t", "indu_t",
]
CONDITION_COLS = ["bat_class", "bat_cc", "soc_status", "tr_method"]


def make_causal_sensor_features(df: pd.DataFrame,
                                sensor_cols: list[str] | None = None,
                                windows_s: tuple[int, ...] = (30, 60)) -> pd.DataFrame:
    """Return sample_id + numeric model features, in EXACT input row order.

    Window mean includes current observation and existing prior observations within
    [t-window, t]. Missing data is ignored in window mean, but retained as NaN
    if a window has no valid measurement. d/dt from immediately preceding
    *observed* row is NaN at experiment start and when either reading is missing.
    """
    if sensor_cols is None:
        sensor_cols = DEFAULT_SENSORS
    needed = ["sample_id", "experiment_id", "relative_time_s", *sensor_cols, *CONDITION_COLS]
    missing = set(needed).difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.sample_id.duplicated().any():
        raise ValueError("Duplicate sample_id in provided split")
    if df.duplicated(["experiment_id", "relative_time_s"]).any():
        raise ValueError("Duplicate experiment/timestamp")
    windows_s = tuple(sorted(set(int(w) for w in windows_s)))
    if any(w <= 0 for w in windows_s):
        raise ValueError("windows_s must be positive")
    chunks = []
    # sort=False preserves group arrival order; final restore original row order
    working = df.copy()
    working["_original_position"] = np.arange(len(working))
    for _, part in working.groupby("experiment_id", sort=False):
        part = part.sort_values("relative_time_s", kind="mergesort")
        times = part.relative_time_s.to_numpy(dtype=float)
        if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("relative_time_s must be finite, unique and increasing within each experiment")
        dprev = np.r_[np.nan, np.diff(times)]
        out = {"sample_id": part.sample_id.to_numpy(),
               "_original_position": part._original_position.to_numpy(),
               "relative_time_s": times,
               "dt_prev_s": dprev}
        for col in CONDITION_COLS:
            out[col] = part[col].to_numpy()
        for col in sensor_cols:
            values = part[col].to_numpy(dtype=float)
            finite = np.isfinite(values)
            out[col] = values
            prev = np.r_[np.nan, values[:-1]]
            out[f"{col}_rate_prev"] = np.divide(values - prev, dprev,
                                                   out=np.full(len(part), np.nan),
                                                   where=np.isfinite(dprev) & (dprev > 0))
            for w in windows_s:
                lo = np.searchsorted(times, times - w, side="left")
                valid_values = np.where(finite, values, 0.0)
                cumul = np.r_[0.0, np.cumsum(valid_values)]
                counts = np.r_[0, np.cumsum(finite.astype(np.int64))]
                sums = cumul[1:] - cumul[lo]
                cnt = counts[1:] - counts[lo]
                out[f"{col}_mean_{w}s"] = np.divide(sums, cnt, out=np.full(len(part), np.nan),
                                                       where=cnt > 0)
        chunks.append(pd.DataFrame(out))
    if not chunks:
        return pd.DataFrame(columns=["sample_id"])
    return (pd.concat(chunks, ignore_index=True)
            .sort_values("_original_position")
            .drop(columns="_original_position")
            .reset_index(drop=True))
