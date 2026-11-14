"""Train sensor-only HGB component on Train+Validation for FINAL 50:50 ensemble.

Matches experiments/sensor_cv_example.py exactly (feature recipe and HGB params).
NEVER reads Test CSV; this SENSOR model cannot be submitted alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from cv.common import load_labeled
from features.causal_sensor_features import make_causal_sensor_features

LABELS = [1, 2, 3, 4, 5, 6]


def train_sensor(data_dir: Path, out_dir: Path, max_iter: int = 75, overwrite: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "sensor_model.joblib"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists: preserving previously trained model")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    df = load_labeled(data_dir / "train" / "train.csv",
                      data_dir / "validation" / "validation.csv")
    if df["experiment_id"].nunique() != 97 or len(df) != 43722:
        print(f"WARNING: labeled data shape differs from known competition data: "
              f"rows={len(df)}, experiments={df['experiment_id'].nunique()}", flush=True)
    X = make_causal_sensor_features(df).drop(columns="sample_id")
    columns = list(X.columns)
    # Same model settings as the pre-existing five-fold sensor comparison.
    model = HistGradientBoostingClassifier(
        max_iter=max_iter, max_leaf_nodes=31, learning_rate=0.07,
        class_weight="balanced", l2_regularization=1.0,
        early_stopping=False, random_state=42,
    )
    model.fit(X, df["label"])
    if model.classes_.tolist() != LABELS:
        raise ValueError(f"Missing classes: {model.classes_}")
    payload = {
        "format": "unid-sensor-hgb-final-v1", "model": model,
        "columns": columns, "labels": LABELS,
        "n_train_rows": len(df), "n_train_experiments": int(df.experiment_id.nunique()),
        "max_iter": max_iter, "sklearn_version": sklearn.__version__,
    }
    joblib.dump(payload, target, compress=3)
    (out_dir / "ensemble_config.json").write_text(json.dumps({
        "format": "unid-ensemble-v1", "sensor_model": "sensor_model.joblib",
        "fusion_dir": "fusion", "sensor_weight": 0.5,
        "fusion_weight": 0.5, "labels": LABELS,
        "note": "Weights fixed in advance from labeled OOF; Test was not used for tuning.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SAVED {target} ({target.stat().st_size / 1048576:.2f} MiB)", flush=True)
    print(f"Training experiments={df.experiment_id.nunique()}, rows={len(df)}, "
          f"features={len(columns)}, classes={model.classes_.tolist()}", flush=True)
    print("No Test used. No Test inference performed.", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="weights/ensemble_v1")
    ap.add_argument("--max-iter", type=int, default=75)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    train_sensor(Path(a.data_dir).expanduser(), Path(a.out_dir), a.max_iter, a.overwrite)


if __name__ == "__main__":
    main()
