"""Official Uni-DTHON interface. No fitting, no test cache, no future observations.

python test.py --input PATH --image_root DIR --output PATH
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from model import (GatedFusion, SensorNormalizer, build_sensor_features,
                   encode_images, image_paths, make_backbone, set_determinism)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--image_root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    set_determinism(42)
    # For repeatability with official permutation / experiment-splitting checks.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Path(__file__).resolve().parent / "weights" / "model.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {weights}. Place trained model here.")
    bundle = torch.load(weights, map_location="cpu", weights_only=True)
    if bundle.get("model_type") != "uni_dthon_convnext_tiny_gated_v1":
        raise ValueError("Wrong checkpoint model_type")
    backbone = make_backbone()
    backbone.load_state_dict(bundle["backbone_state"], strict=True)
    fusion = GatedFusion(bundle["sensor_dim"], dropout=bundle["dropout"])
    fusion.load_state_dict(bundle["fusion_state"], strict=True)
    fusion = fusion.eval().to(device)
    norm = SensorNormalizer.from_dict(bundle["normalizer"])

    frame = pd.read_csv(args.input, dtype={"sample_id": "string", "experiment_id": "string"})
    X = norm.transform(build_sensor_features(frame))
    paths = image_paths(frame, args.image_root)
    image_embed = encode_images(backbone, paths, device, batch_size=64,
                                num_workers=2, size=bundle["image_size"])
    backbone.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    preds = []
    with torch.inference_mode():
        for start in range(0, len(frame), 2048):
            end = start + 2048
            a, _, _, _ = fusion(torch.from_numpy(X[start:end]).to(device),
                                torch.from_numpy(image_embed[start:end]).to(device))
            preds.extend((a.argmax(dim=1).cpu().numpy() + 1).tolist())
    output = pd.DataFrame({"sample_id": frame["sample_id"].astype(str),
                           "label": np.asarray(preds, dtype=int)})
    if list(output.columns) != ["sample_id", "label"] or not output["label"].isin(range(1, 7)).all():
        raise AssertionError("Submission format invalid")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"[SAVED] {args.output} rows={len(output)} device={device}")

if __name__ == "__main__":
    main()
