"""Small PyTorch Dataset connecting preprocessed sensors to the right image.

Each item is {'image', 'sensor', 'sample_id', ('label' if labeled)}.
No caching of labels/test data. No fitting or scaling happens in Dataset.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from features.image_transform import ThermalImageTransform
from features.sensor_preprocess import SensorPreprocessor


class ImageSensorDataset:
    def __init__(self, frame: pd.DataFrame, preprocessor: SensorPreprocessor,
                 image_root: str | Path | None = None, image_root_col: str = "_image_root",
                 image_transform: ThermalImageTransform | None = None):
        """For combined Train+Validation, provide `_image_root` per input row.

        For competition Test CLI, pass image_root=... and omit `_image_root`.
        Test predictions align with frame's ORIGINAL row order.
        """
        if image_root is None and image_root_col not in frame.columns:
            raise ValueError("Pass image_root or provide a per-row '_image_root' column")
        if "image_path" not in frame.columns:
            raise ValueError("Missing image_path")
        self.frame = frame.reset_index(drop=True).copy()
        self.x_sensor = preprocessor.transform(self.frame)
        self.image_transform = image_transform or ThermalImageTransform()
        self.image_root = Path(image_root) if image_root is not None else None
        self.image_root_col = image_root_col
        self.has_label = "label" in self.frame.columns
        self.ids = self.frame["sample_id"].tolist()
        if self.has_label:
            labels = pd.to_numeric(self.frame.label, errors="raise").to_numpy(dtype=np.int64)
            if not np.isin(labels, [1, 2, 3, 4, 5, 6]).all():
                raise ValueError("Labels must be in 1..6")
            self.labels = labels - 1  # PyTorch CE target 0..5
        else:
            self.labels = None

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        # Torch import occurs only when a sample is actually requested.
        import torch
        row = self.frame.iloc[idx]
        root = self.image_root if self.image_root is not None else row[self.image_root_col]
        result = {
            "image": torch.from_numpy(self.image_transform.load(root, row.image_path)),
            "sensor": torch.from_numpy(self.x_sensor[idx].copy()),
            "sample_id": self.ids[idx],
        }
        if self.has_label:
            result["label"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return result
