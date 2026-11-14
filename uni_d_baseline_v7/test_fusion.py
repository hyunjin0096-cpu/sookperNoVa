"""Competition-compatible deterministic inference. DOES NOT train or fit anything.

Usage:
  python test_fusion.py --input TEST_CSV --image_root IMAGE_DIR --output SUBMISSION_CSV

After validation: cp test.py test_baseline.py && cp test_fusion.py test.py
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from features.image_transform import ThermalImageTransform
from features.sensor_preprocess import SensorPreprocessor
from features.multimodal_dataset import ImageSensorDataset
from fusion_model import FusionNet


def predict(frame: pd.DataFrame, image_root: Path, weights_dir: Path,
            device: torch.device, batch_size: int = 1) -> tuple[np.ndarray, np.ndarray]:
    if "label" in frame:
        # Input can be labeled for self-testing; no label is accessed by the model.
        frame = frame.drop(columns=["label"])
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("input CSV must have unique nonmissing sample_id")
    config = json.loads((weights_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("format") != "unid-fusion-v1.1" or config.get("labels") != [1, 2, 3, 4, 5, 6]:
        raise ValueError("Unexpected model format / label ordering")
    tfm_config = config["image_transform"]
    transform = ThermalImageTransform(
        size=int(tfm_config["size"]), mode=str(tfm_config["mode"]),
        mean=tuple(tfm_config["mean"]), std=tuple(tfm_config["std"]),
        pad_rgb=tuple(tfm_config["pad_rgb"]),
    )
    prep = SensorPreprocessor.load(weights_dir / "sensor_preprocess.json")
    if int(config["sensor_dim"]) != len(prep.feature_columns):
        raise ValueError("Model sensor_dim mismatches saved preprocessor")
    model = FusionNet(sensor_dim=int(config["sensor_dim"]), pretrained_backbone=False)
    # Saved weights are OUR model only, not an arbitrary pickled model object.
    state = torch.load(weights_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    ds = ImageSensorDataset(frame.reset_index(drop=True), prep,
                            image_root=image_root, image_transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    result = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            sensor = batch["sensor"].to(device)
            logits = model(image, sensor)
            result.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    probabilities = np.concatenate(result, axis=0) if result else np.empty((0, 6), np.float32)
    if len(probabilities) != len(frame):
        raise AssertionError("Prediction row count changed")
    pred_labels = probabilities.argmax(axis=1).astype(np.int64) + 1
    return pred_labels, probabilities


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--weights-dir", default=None,
                   help="default: weights/ next to test_fusion.py (final: test.py)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    weights_dir = Path(args.weights_dir).resolve() if args.weights_dir else here / "weights"
    frame = pd.read_csv(args.input)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    # Stabilize inference; image input is not randomized and batch size is 1.
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    labels, _ = predict(frame, Path(args.image_root), weights_dir, device, batch_size=1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": frame["sample_id"], "label": labels}).to_csv(output, index=False)
    print(f"Saved {len(labels)} predictions -> {output}", flush=True)


if __name__ == "__main__":
    main()
