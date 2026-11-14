"""Offline, row-order-preserving Uni-DTHON 50:50 ensemble inference.

Official CLI (after promoting THIS file to test.py):
  python test.py --input TEST.csv --image_root TEST_DIR --output submission.csv

Uses ONLY rows given in --input; NO training, NO fitting, NO future rows,
NO external weights, NO reading competition Test outside supplied input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from features.causal_sensor_features import make_causal_sensor_features
from test_fusion import predict as predict_fusion

LABELS = [1, 2, 3, 4, 5, 6]
PROBS = [f"prob_{i}" for i in LABELS]


def load_ensemble(weights_dir: Path):
    config = json.loads((weights_dir / "ensemble_config.json").read_text(encoding="utf-8"))
    if config.get("format") != "unid-ensemble-v1" or config.get("labels") != LABELS:
        raise ValueError("Unexpected ensemble config/labels")
    alpha = float(config.get("sensor_weight"))
    beta = float(config.get("fusion_weight"))
    if not np.isclose(alpha + beta, 1.0) or min(alpha, beta) <= 0:
        raise ValueError("Invalid weights. Both modalities must contribute")
    sensor_path = weights_dir / config["sensor_model"]
    fusion_dir = weights_dir / config["fusion_dir"]
    if not all((fusion_dir / n).is_file() for n in ("model.pt", "sensor_preprocess.json", "config.json")):
        raise FileNotFoundError(f"Missing Fusion checkpoint in {fusion_dir}")
    sensor_saved = joblib.load(sensor_path)  # ONLY trusted self-trained artifact
    if sensor_saved.get("format") != "unid-sensor-hgb-final-v1":
        raise ValueError("Unknown sensor checkpoint format")
    if sensor_saved.get("labels") != LABELS:
        raise ValueError("Unexpected sensor label map")
    model = sensor_saved["model"]
    if list(model.classes_) != LABELS:
        raise ValueError("Sensor model classes do not match labels")
    return config, sensor_saved, fusion_dir


def predict(frame: pd.DataFrame, image_root: Path, weights_dir: Path,
            device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    if len(frame) == 0:
        raise ValueError("Cannot predict an empty CSV")
    if "sample_id" not in frame or frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("Missing / duplicate sample_id")
    # Dropping labels for a diagnostic labeled input ensures labels are never features.
    clean = frame.drop(columns="label", errors="ignore").copy()
    config, saved, fusion_dir = load_ensemble(weights_dir)
    sensor_features = make_causal_sensor_features(clean).drop(columns="sample_id")
    cols = list(saved["columns"])
    if set(cols) != set(sensor_features.columns):
        raise ValueError("Sensor feature schema mismatch")
    sensor_probs_raw = saved["model"].predict_proba(sensor_features[cols])
    sensor_probs = np.zeros((len(frame), 6), dtype=np.float64)
    for j, label in enumerate(saved["model"].classes_):
        sensor_probs[:, int(label) - 1] = sensor_probs_raw[:, j]

    # Trained Fusion checkpoint (includes RGB thermal images); batch_size 1.
    _, fusion_probs = predict_fusion(clean, image_root, fusion_dir, device, batch_size=1)
    if sensor_probs.shape != fusion_probs.shape or sensor_probs.shape != (len(frame), 6):
        raise ValueError("Modality prediction shape mismatch")
    if not np.isfinite(sensor_probs).all() or not np.isfinite(fusion_probs).all():
        raise ValueError("Non-finite probabilities")
    p = float(config["sensor_weight"]) * sensor_probs + float(config["fusion_weight"]) * fusion_probs
    if not np.allclose(p.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Ensemble probabilities must sum to 1")
    labels = np.argmax(p, axis=1).astype(np.int64) + 1
    return labels, p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights-dir", default=None,
                        help="default: weights/ensemble_v1 relative to this script")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = parser.parse_args()
    source_dir = Path(__file__).resolve().parent
    weights_dir = (Path(a.weights_dir).expanduser().resolve() if a.weights_dir
                   else source_dir / "weights" / "ensemble_v1")
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available()
                          else "cpu" if a.device == "auto" else a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    csv = pd.read_csv(a.input)
    predictions, _ = predict(csv, Path(a.image_root), weights_dir, device)
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": csv.sample_id, "label": predictions}).to_csv(path, index=False)
    print(f"ENSEMBLE SUBMISSION SAVED {len(csv)} rows -> {path}", flush=True)


if __name__ == "__main__":
    main()
