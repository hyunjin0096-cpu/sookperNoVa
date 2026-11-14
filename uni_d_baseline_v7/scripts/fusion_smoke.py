"""Synthetic integration smoke: preprocessor -> image dataset -> ResNet18 fusion -> backward -> save/load.
No competition data and no actual model artifacts produced.
"""
from __future__ import annotations
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from features.causal_sensor_features import DEFAULT_SENSORS
from features.sensor_preprocess import SensorPreprocessor
from features.image_transform import ThermalImageTransform
from features.multimodal_dataset import ImageSensorDataset
from fusion_model import FusionNet
from test_fusion import predict


def build_mock_data(root: Path):
    (root / "images").mkdir(parents=True)
    rows = []
    for eid in (101, 202):
        for k, t in enumerate((0, 5, 15, 30)):
            sid = f"{eid}_{k}"
            relative = f"images/{sid}.png"
            Image.new("RGB", (40 + k * 2, 72), (20 + k * 40, 70 + k * 20, 110)).save(root / relative)
            row = dict(sample_id=sid, experiment_id=eid, relative_time_s=t,
                       image_path=relative, bat_class=1, bat_cc=4.8,
                       soc_status=1, tr_method=1, label=k + 1)
            for j, col in enumerate(DEFAULT_SENSORS):
                row[col] = float(10 + j + t * .1)
            row["indu_t"] = np.nan if eid == 202 else float(20 + t)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    torch.manual_seed(2026)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        frame = build_mock_data(root)
        prep = SensorPreprocessor().fit(frame)
        transform = ThermalImageTransform(size=64)
        ds = ImageSensorDataset(frame, prep, image_root=root, image_transform=transform)
        batch = next(iter(DataLoader(ds, batch_size=2, shuffle=False)))
        model = FusionNet(sensor_dim=len(prep.feature_columns))
        logits = model(batch["image"], batch["sensor"])
        assert logits.shape == (2, 6)
        loss = torch.nn.CrossEntropyLoss()(logits, batch["label"])
        loss.backward()
        assert torch.isfinite(loss).item()
        weights_dir = root / "weights"
        weights_dir.mkdir()
        torch.save(model.state_dict(), weights_dir / "model.pt")
        prep.save(weights_dir / "sensor_preprocess.json")
        import json
        config = {"format": "unid-fusion-v1.1", "sensor_dim": len(prep.feature_columns),
                  "image_transform": transform.config(), "labels": [1, 2, 3, 4, 5, 6]}
        (weights_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        model.eval()
        expected = []
        with torch.inference_mode():
            for batch in DataLoader(ds, batch_size=1):
                expected.append(torch.softmax(model(batch["image"], batch["sensor"]), 1).numpy())
        expected = np.concatenate(expected)
        pred, prob = predict(frame, root, weights_dir, torch.device("cpu"), batch_size=1)
        np.testing.assert_allclose(prob, expected, rtol=1e-5, atol=1e-6)
        assert np.array_equal(pred, expected.argmax(axis=1) + 1)
        print("FUSION SYNTHETIC SMOKE OK — tensors, gradient, save/load, prediction matched", flush=True)


if __name__ == "__main__":
    main()
