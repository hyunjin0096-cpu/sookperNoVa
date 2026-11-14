"""Uni-DTHON v1.1: fold-fitted, causal sensor preprocessing.

Shared by training and inference. Never fits on validation/test at transform time.
Use fit(outer_train) for CV; fit(all_labeled_train) only for the final model.

Output: float32 ndarray [N, D] in exact original CSV row order.
All sample/experiment identifiers and labels are excluded from model features.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.causal_sensor_features import (
    CONDITION_COLS,
    DEFAULT_SENSORS,
    make_causal_sensor_features,
)


# Known categories from the competition schema. No categories are learned from Test.
CATEGORIES = {
    "bat_class": (1, 2, 3),
    "soc_status": (1, 2),
    "tr_method": (1, 2),
}
LOG1P_COLUMNS = ("relative_time_s", "dt_prev_s", "bat_cc")
VERSION = "unid-sensor-v1.1"


class SensorPreprocessor:
    """Causal time features + robust scaling + missing flags + one-hot conditions.

    Each numeric feature gets a dedicated missing flag, even when all training
    values are present; this keeps the feature schema stable for each Fold.
    Missing-value fill/center/scale are learned ONLY in fit().
    """

    def __init__(self, windows_s: tuple[int, ...] = (30, 60), clip: float = 10.0):
        windows = tuple(sorted(set(int(w) for w in windows_s)))
        if not windows or any(w <= 0 for w in windows):
            raise ValueError("windows_s must contain strictly positive seconds")
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError("clip must be positive and finite")
        self.windows_s = windows
        self.clip = float(clip)
        self.numeric_columns: list[str] | None = None
        self.feature_columns: list[str] | None = None
        self.medians: dict[str, float] | None = None
        self.scales: dict[str, float] | None = None

    @property
    def is_fitted(self) -> bool:
        return self.numeric_columns is not None and self.medians is not None

    def _build_feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        # Existing CV implementation provides time-causal, input-order invariant
        # features based on real seconds, never 'last N rows'.
        causal = make_causal_sensor_features(
            frame, sensor_cols=DEFAULT_SENSORS, windows_s=self.windows_s
        )
        for col, categories in CATEGORIES.items():
            observed = pd.to_numeric(causal[col], errors="coerce").dropna().unique()
            unseen = sorted(set(observed).difference(categories))
            if unseen:
                raise ValueError(f"Unknown {col} categories: {unseen}")
        numeric_cols = [
            col for col in causal.columns
            if col != "sample_id" and col not in CATEGORIES
        ]
        X = causal[numeric_cols].apply(pd.to_numeric, errors="coerce").copy()
        for col in LOG1P_COLUMNS:
            if col not in X:
                raise ValueError(f"Required continuous feature missing: {col}")
            # An invalid negative time/capacity is a data-contract error; missing
            # values (such as the initial dt_prev_s) remain NaN.
            invalid = X[col].notna() & (X[col] < 0)
            if invalid.any():
                raise ValueError(f"Negative values in {col}; cannot log1p")
            X[col] = np.log1p(X[col])
        X = X.replace([np.inf, -np.inf], np.nan)
        for col, categories in CATEGORIES.items():
            for value in categories:
                X[f"{col}__{value}"] = (
                    pd.to_numeric(causal[col], errors="coerce") == value
                ).astype(np.float32)
        return X

    def fit(self, train_frame: pd.DataFrame) -> "SensorPreprocessor":
        """Fit only on the current training partition (not validation/test)."""
        X = self._build_feature_frame(train_frame)
        numeric_cols = [c for c in X.columns if c not in self._categorical_names()]
        self.numeric_columns = numeric_cols
        self.medians = {}
        self.scales = {}
        for col in numeric_cols:
            values = X[col].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                median, iqr = 0.0, 1.0
            else:
                median = float(np.median(values))
                q25, q75 = np.percentile(values, [25.0, 75.0])
                iqr = float(q75 - q25)
                if not np.isfinite(iqr) or iqr < 1e-9:
                    iqr = 1.0
            self.medians[col] = float(median)
            self.scales[col] = float(iqr)
        self.feature_columns = (
            [f"{c}__scaled" for c in numeric_cols]
            + [f"{c}__missing" for c in numeric_cols]
            + self._categorical_names()
        )
        return self

    @staticmethod
    def _categorical_names() -> list[str]:
        return [f"{key}__{cat}" for key, cats in CATEGORIES.items() for cat in cats]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Transform without fitting. Returns (N,D) float32 in input row order."""
        if not self.is_fitted:
            raise RuntimeError("SensorPreprocessor must be fitted or loaded before transform")
        X = self._build_feature_frame(frame)
        required = self.numeric_columns + self._categorical_names()
        missing = [col for col in required if col not in X]
        if missing:
            raise ValueError(f"Feature schema mismatch; missing: {missing}")
        values = X[self.numeric_columns].to_numpy(dtype=np.float64)
        miss = (~np.isfinite(values)).astype(np.float32)
        center = np.asarray([self.medians[c] for c in self.numeric_columns], dtype=np.float64)
        scale = np.asarray([self.scales[c] for c in self.numeric_columns], dtype=np.float64)
        # Every NaN/non-finite value becomes the train-fold median (scaled zero).
        filled = np.where(np.isfinite(values), values, center[None, :])
        scaled = np.clip((filled - center[None, :]) / scale[None, :], -self.clip, self.clip)
        categorical = X[self._categorical_names()].to_numpy(dtype=np.float32)
        result = np.concatenate([scaled.astype(np.float32), miss, categorical], axis=1)
        if not np.isfinite(result).all():
            raise RuntimeError("Non-finite post-processed sensor feature")
        if result.shape[1] != len(self.feature_columns):
            raise RuntimeError("Unexpected output dimension")
        return result

    def fit_transform(self, train_frame: pd.DataFrame) -> np.ndarray:
        return self.fit(train_frame).transform(train_frame)

    def state_dict(self) -> dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessing")
        return {
            "version": VERSION,
            "windows_s": list(self.windows_s),
            "clip": self.clip,
            "numeric_columns": list(self.numeric_columns),
            "feature_columns": list(self.feature_columns),
            "medians": dict(self.medians),
            "scales": dict(self.scales),
            "sensors": list(DEFAULT_SENSORS),
            "conditions": list(CONDITION_COLS),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.state_dict(), ensure_ascii=False,
                                     indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SensorPreprocessor":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("version") != VERSION:
            raise ValueError(f"Preprocessor version mismatch: {state.get('version')}")
        if state.get("sensors") != DEFAULT_SENSORS or state.get("conditions") != CONDITION_COLS:
            raise ValueError("Sensor/condition schema mismatch: use matching code")
        instance = cls(tuple(state["windows_s"]), float(state["clip"]))
        instance.numeric_columns = list(state["numeric_columns"])
        instance.feature_columns = list(state["feature_columns"])
        instance.medians = {k: float(v) for k, v in state["medians"].items()}
        instance.scales = {k: float(v) for k, v in state["scales"].items()}
        expected = (
            [f"{c}__scaled" for c in instance.numeric_columns]
            + [f"{c}__missing" for c in instance.numeric_columns]
            + instance._categorical_names()
        )
        if instance.feature_columns != expected:
            raise ValueError("Corrupt preprocessor feature order")
        if set(instance.numeric_columns) != set(instance.medians) or set(instance.numeric_columns) != set(instance.scales):
            raise ValueError("Corrupt preprocessor statistics")
        if any(not np.isfinite(s) or s <= 0 for s in instance.scales.values()):
            raise ValueError("Corrupt preprocessing scale")
        return instance
