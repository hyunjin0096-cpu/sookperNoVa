"""Gap-view robustness sampling using ONLY the publicly specified {5,10,15}s gaps.

Works on original 5s labeled train/validation observations. No Test statistics.
Do NOT use these thinned data as the primary 5-second OOF metric.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_gap_view(df: pd.DataFrame, seed: int = 2026,
                  step_probabilities: tuple[float, float, float] = (1/3, 1/3, 1/3)) -> pd.DataFrame:
    probs = np.array(step_probabilities, dtype=float)
    if probs.shape != (3,) or np.any(probs < 0) or not np.isclose(probs.sum(), 1):
        raise ValueError("step_probabilities must be 3 nonnegative values summing to 1")
    if not {"experiment_id", "relative_time_s", "sample_id"}.issubset(df.columns):
        raise ValueError("Input requires experiment_id, relative_time_s, sample_id")
    if df.sample_id.duplicated().any():
        raise ValueError("Duplicated sample IDs")
    # Deterministic per-experiment RNG by sorting IDs and using a fixed generator.
    rng = np.random.default_rng(seed)
    selected = []
    for _, group in df.groupby("experiment_id", sort=True):
        g = group.sort_values("relative_time_s")
        time = g.relative_time_s.to_numpy(dtype=float)
        if len(time) < 1 or (np.diff(time) <= 0).any():
            raise ValueError("Experiment contains duplicate/non-increasing timestamps")
        # Starts at first observation and keeps later rows near t+5/10/15s;
        # if that time is absent, chooses the first observation at/after target.
        positions = [0]
        pos = 0
        while pos < len(g) - 1:
            target = time[pos] + float(rng.choice([5, 10, 15], p=probs))
            next_pos = max(pos + 1, int(np.searchsorted(time, target, side="left")))
            if next_pos >= len(g):
                break
            positions.append(next_pos)
            pos = next_pos
        selected.append(g.iloc[positions])
    return pd.concat(selected, ignore_index=True)
