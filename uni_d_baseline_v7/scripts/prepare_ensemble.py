"""Stage existing FINAL FusionNet checkpoint alongside newly trained sensor HGB.

Copies the final FusionNet model and fitted scaler; never retrains or alters either.
--fp16-storage reduces file size but introduces small numerical changes and
requires new verification before submitting. Default: exact weight copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while part := f.read(1024 * 1024):
            h.update(part)
    return h.hexdigest()


def prepare(source: Path, dest: Path, fp16_storage: bool = False, overwrite: bool = False):
    source = source.resolve()
    dest = dest.resolve()
    if source == dest or source == dest / "fusion":
        raise ValueError("source and destination overlap")
    needed = ("model.pt", "sensor_preprocess.json", "config.json")
    for name in needed:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Expected final Fusion weight: {source / name}")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("format") != "unid-fusion-v1.1" or config.get("fold") is not None:
        raise ValueError("Not a FINAL full-data Fusion checkpoint (fold must be null)")
    target = dest / "fusion"
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} exists. Refusing overwrite.")
        # Safe and predictable: overwrite only the three known files, no recursive delete.
    target.mkdir(parents=True, exist_ok=True)
    for name in needed:
        src = source / name
        dst = target / name
        if name == "model.pt" and fp16_storage:
            import torch
            state = torch.load(src, map_location="cpu", weights_only=True)
            half = {k: (v.to(dtype=torch.float16) if v.is_floating_point() else v)
                    for k, v in state.items()}
            torch.save(half, dst)
        else:
            shutil.copy2(src, dst)
        print(f"STAGED {dst} ({dst.stat().st_size / 1048576:.2f} MiB)", flush=True)
        if not fp16_storage or name != "model.pt":
            if sha256(src) != sha256(dst):
                raise AssertionError(f"Copied file mismatch: {name}")
    if fp16_storage:
        print("WARNING: fp16-storage changes weights slightly. Re-run inference and checks.")
    if (dest / "sensor_model.joblib").is_file():
        print("READY: both fusion and sensor weights present")
    else:
        print("Sensor model not found. Run train_sensor_ensemble.py next.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="weights/fusion_v1")
    ap.add_argument("--dest", default="weights/ensemble_v1")
    ap.add_argument("--fp16-storage", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    prepare(Path(args.source), Path(args.dest), args.fp16_storage, args.overwrite)


if __name__ == "__main__":
    main()
