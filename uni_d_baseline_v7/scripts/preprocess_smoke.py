"""Read a few provided rows on the team's competition server; verify full pipeline.

No data or images are written anywhere. Fits a throwaway preprocessor on small
TRAIN-only groups strictly as a functionality check, NOT a usable saved model.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features.image_transform import ThermalImageTransform
from features.sensor_preprocess import SensorPreprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="e.g. ~/data/battery_data_light")
    parser.add_argument("--samples-per-exp", type=int, default=8)
    args = parser.parse_args()
    if args.samples_per_exp < 2:
        raise ValueError("--samples-per-exp must be >=2")
    root = Path(args.data_dir).expanduser()
    train = pd.read_csv(root / "train" / "train.csv")
    valid = pd.read_csv(root / "validation" / "validation.csv")
    # For this smoke test alone, take FIRST n rows of three experiments each.
    # In real CV, fit only on outer/inner training groups.
    tr = (train.sort_values(["experiment_id", "relative_time_s"])
          .groupby("experiment_id", group_keys=False).head(args.samples_per_exp))
    tr = tr[tr.experiment_id.isin(tr.experiment_id.unique()[:3])].reset_index(drop=True)
    va = (valid.sort_values(["experiment_id", "relative_time_s"])
          .groupby("experiment_id", group_keys=False).head(args.samples_per_exp))
    va = va[va.experiment_id.isin(va.experiment_id.unique()[:2])].reset_index(drop=True)
    prep = SensorPreprocessor().fit(tr)
    tr_feat = prep.transform(tr)
    va_feat = prep.transform(va)
    assert tr_feat.shape[0] == len(tr) and va_feat.shape[0] == len(va)
    assert tr_feat.shape[1] == va_feat.shape[1] == len(prep.feature_columns)
    assert np.isfinite(tr_feat).all() and np.isfinite(va_feat).all()
    tfm = ThermalImageTransform(size=160, mode="pad")
    example = tfm.load(root / "train", tr.image_path.iloc[0])
    assert example.shape == (3, 160, 160)
    # Shuffle only: the resulting sensor feature for each sample must match.
    shuffled = va.sample(frac=1, random_state=2026)
    reference = dict(zip(va.sample_id, va_feat))
    for sid, feat in zip(shuffled.sample_id, prep.transform(shuffled)):
        np.testing.assert_array_equal(feat, reference[sid])
    print("PREPROCESS SMOKE OK")
    print("train rows", len(tr), "validation rows", len(va))
    print("sensor feature D", tr_feat.shape[1], "image shape", example.shape)
    print("shuffle-invariant: OK; train-fitted validation transform: OK")
    print("Note: this smoke test did not train a model or create artifacts.")


if __name__ == "__main__":
    main()
