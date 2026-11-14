"""Reproduce final Uni-DTHON image+sensor ensemble weights from Train+Validation.

Default retrains final Fusion NN and sensor HGB using only labeled data.
To reuse a checkpoint already trained on all 97 experiments, pass
--reuse-fusion-final weights/fusion_v1. Never read Test for training.

After choosing this model, this script may become the organizer-facing train.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from train_sensor_ensemble import train_sensor
from scripts.prepare_ensemble import prepare


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="weights/ensemble_v1")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--reuse-fusion-final", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir)
    if a.reuse_fusion_final is None:
        cmd = [sys.executable, "-m", "experiments.train_fusion", "--data-dir", a.data_dir,
               "--mode", "final", "--epochs", str(a.epochs),
               "--batch-size", str(a.batch_size), "--num-workers", str(a.num_workers),
               "--out-dir", str(out / "fusion")]
        if a.overwrite:
            cmd.append("--overwrite")
        print("Training final Fusion NN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    else:
        print("Reusing verified FINAL Fusion checkpoint:", a.reuse_fusion_final, flush=True)
        prepare(Path(a.reuse_fusion_final), out, overwrite=a.overwrite)
    train_sensor(Path(a.data_dir).expanduser(), out, max_iter=75, overwrite=a.overwrite)
    for path in (out / "fusion" / "model.pt", out / "fusion" / "config.json",
                 out / "fusion" / "sensor_preprocess.json", out / "sensor_model.joblib",
                 out / "ensemble_config.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing final artifact: {path}")
    print(f"FINAL ENSEMBLE WEIGHTS READY: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
