"""Aggregate-only error analysis for Uni-DTHON ConvNeXt + Sensor MLP + Gated Fusion.

Run from the directory containing model.py:
python analyze_convnext_errors.py --data-root /home/ubuntu/data/battery_data_light \
  --checkpoint weights/model.pt --out-dir /home/ubuntu/uni_dthon_private_diagnostics

No model fitting, no Test data, no per-sample private files are saved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from model import (IMAGE_MEAN, IMAGE_STD, GatedFusion, SensorNormalizer,
                   build_sensor_features, encode_images, image_paths,
                   make_backbone, set_determinism)

STAGES = list(range(1, 7))


def summarize(y: np.ndarray, preds: np.ndarray):
    p, r, f1, n = precision_recall_fscore_support(y, preds, labels=STAGES, zero_division=0)
    return pd.DataFrame({"stage": STAGES, "support": n,
                         "precision": p, "recall": r, "f1": f1})


def overall(y, pred):
    return {"macro_f1": round(float(f1_score(y, pred, labels=STAGES,
                                            average="macro", zero_division=0)), 6),
            "accuracy": round(float(accuracy_score(y, pred)), 6)}


def predict(net, features, embeds, device, bs):
    net.eval()
    pred, sensor, image, gate = [], [], [], []
    with torch.inference_mode():
        for i in range(0, len(features), bs):
            j = min(i + bs, len(features))
            x = torch.from_numpy(features[i:j]).to(device)
            im = torch.from_numpy(embeds[i:j]).to(device)
            a, b, c, g = net(x, im)
            pred.append(a.argmax(1).cpu().numpy() + 1)
            sensor.append(b.argmax(1).cpu().numpy() + 1)
            image.append(c.argmax(1).cpu().numpy() + 1)
            gate.append(g.mean(1).cpu().numpy())
    return tuple(np.concatenate(z) for z in (pred, sensor, image, gate))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/model.pt"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/home/ubuntu/uni_dthon_private_diagnostics"))
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--skip-black-test", action="store_true")
    args = parser.parse_args()

    set_determinism(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}", flush=True)
    val_path = args.data_root / "validation" / "validation.csv"
    val = pd.read_csv(val_path, dtype={"sample_id": "string", "experiment_id": "string"})
    if "label" not in val:
        raise ValueError("This diagnostic requires labeled Validation CSV; never run on Test labels.")
    y = val["label"].to_numpy(dtype=int)
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if bundle.get("model_type") != "uni_dthon_convnext_tiny_gated_v1":
        raise ValueError("Unexpected checkpoint type")
    norm = SensorNormalizer.from_dict(bundle["normalizer"])
    x = norm.transform(build_sensor_features(val))
    net = GatedFusion(bundle["sensor_dim"], dropout=bundle["dropout"])
    net.load_state_dict(bundle["fusion_state"], strict=True)
    net = net.eval().to(device)
    backbone = make_backbone()
    backbone.load_state_dict(bundle["backbone_state"], strict=True)
    print(f"[IMAGES] encoding {len(val)} validation images...", flush=True)
    embeds = encode_images(backbone, image_paths(val, args.data_root / "validation"),
                           device, args.image_batch_size, args.workers,
                           bundle["image_size"])
    preds, spreds, ipreds, gates = predict(net, x, embeds, device, args.batch_size)

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_scores = {name: overall(y, p) for name, p in
                    (("fusion", preds), ("sensor_aux_head", spreds), ("image_aux_head", ipreds))}
    print("\n===== HEAD PERFORMANCE =====")
    for k, v in model_scores.items():
        print(f"{k:20s} F1={v['macro_f1']:.4f} accuracy={v['accuracy']:.4f}")
    score_table = pd.DataFrame([{"head": k, **v} for k, v in model_scores.items()])
    score_table.to_csv(out / "heads_summary.csv", index=False)

    stage_table = summarize(y, preds)
    stage_table.to_csv(out / "class_metrics.csv", index=False)
    cm = confusion_matrix(y, preds, labels=STAGES)
    cm_table = pd.DataFrame(cm, index=[f"actual_{i}" for i in STAGES],
                            columns=[f"pred_{i}" for i in STAGES])
    cm_table.to_csv(out / "confusion_matrix.csv")
    print("\n===== FUSED PER-CLASS =====")
    print(stage_table.round(4).to_string(index=False))
    print("\n===== CONFUSION MATRIX =====")
    print(cm_table.to_string())

    rows = []
    for i in STAGES:
        mask = y == i
        rows.append({"stage": i, "count": int(mask.sum()),
                     "sensor_correct": int(((spreds == y) & mask).sum()),
                     "image_correct": int(((ipreds == y) & mask).sum()),
                     "fusion_correct": int(((preds == y) & mask).sum()),
                     "fusion_rescued_sensor": int(((spreds != y) & (preds == y) & mask).sum()),
                     "fusion_harmed_sensor": int(((spreds == y) & (preds != y) & mask).sum()),
                     "mean_image_gate": round(float(gates[mask].mean()), 5) if mask.any() else None})
    rescue = pd.DataFrame(rows)
    rescue.to_csv(out / "sensor_to_fusion_effect.csv", index=False)
    print("\n===== SENSOR AUX HEAD -> FUSION EFFECT (critical) =====")
    print(rescue.to_string(index=False))

    methods = []
    detailed = []
    vals = val["tr_method"].fillna("missing").astype(str).to_numpy()
    for method in sorted(set(vals)):
        mask = vals == method
        methods.append({"method": method, "count": int(mask.sum()),
                        "fused_f1": overall(y[mask], preds[mask])["macro_f1"],
                        "sensor_aux_f1": overall(y[mask], spreds[mask])["macro_f1"],
                        "image_aux_f1": overall(y[mask], ipreds[mask])["macro_f1"],
                        "mean_gate": round(float(gates[mask].mean()), 5)})
        for actual in (2, 3, 4, 5, 6):
            mask2 = mask & (y == actual)
            if mask2.any():
                detailed.append({"method": method, "actual_stage": actual,
                                 "n": int(mask2.sum()),
                                 "correct": int((preds[mask2] == actual).sum()),
                                 "recall": round(float((preds[mask2] == actual).mean()), 5),
                                 "mean_gate": round(float(gates[mask2].mean()), 5)})
    method_table = pd.DataFrame(methods)
    method_table.to_csv(out / "method_summary.csv", index=False)
    pd.DataFrame(detailed).to_csv(out / "method_stage_recall.csv", index=False)
    print("\n===== BY TR_METHOD =====")
    print(method_table.to_string(index=False))

    # Per-experiment error concentration. Kept PRIVATE in the specified output directory.
    exp_rows = []
    for exp_id, group in val.groupby("experiment_id", sort=False):
        ii = group.index.to_numpy(dtype=int)
        actual, predicted = y[ii], preds[ii]
        labels_present = sorted(set(actual.tolist()))
        mask5 = actual == 5
        exp_rows.append({
            "experiment_id": str(exp_id), "n": len(ii),
            "tr_method": str(group["tr_method"].iloc[0]),
            "stage5_n": int(mask5.sum()),
            "stage5_to3_n": int((mask5 & (predicted == 3)).sum()),
            "stage5_to4_n": int((mask5 & (predicted == 4)).sum()),
            "f1_present_labels": round(float(f1_score(actual, predicted,
                labels=labels_present, average="macro", zero_division=0)), 5),
        })
    exp_df = pd.DataFrame(exp_rows).sort_values("stage5_to3_n", ascending=False)
    exp_df.to_csv(out / "private_experiment_errors.csv", index=False)
    print("\n===== TOP EXPERIMENTS: STAGE 5 -> 3 ERRORS =====")
    print(exp_df.drop(columns="experiment_id").head(8).to_string(index=False))
    print("(Experiment IDs are only in the private CSV, not printed.)")

    focus = []
    for actual, predicted in ((5, 3), (5, 4), (3, 2), (2, 1), (2, 3), (6, 3)):
        mask = (y == actual) & (preds == predicted)
        if not mask.any():
            continue
        focus.append({"actual": actual, "predicted": predicted, "n": int(mask.sum()),
                      "sensor_head_correct": int((spreds[mask] == actual).sum()),
                      "image_head_correct": int((ipreds[mask] == actual).sum()),
                      "sensor_pred_target": int((spreds[mask] == predicted).sum()),
                      "image_pred_target": int((ipreds[mask] == predicted).sum()),
                      "mean_image_gate": round(float(gates[mask].mean()), 5),
                      "method1": int((vals[mask] == '1').sum()),
                      "method2": int((vals[mask] == '2').sum())})
    focus_table = pd.DataFrame(focus)
    focus_table.to_csv(out / "critical_errors.csv", index=False)
    print("\n===== KEY ERRORS, COMPARE HEADS =====")
    print(focus_table.to_string(index=False))

    black_change = None
    if not args.skip_black_test:
        print("\n[BLACK-IMAGE] testing if predictions actually depend on image pixels...", flush=True)
        mean = torch.tensor(IMAGE_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGE_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        im_black = (torch.zeros((1, 3, bundle["image_size"], bundle["image_size"]),
                                device=device) - mean) / std
        backbone.eval().to(device)
        with torch.inference_mode():
            black_embed = backbone(im_black).cpu().numpy().astype(np.float32)
        black_embeds = np.repeat(black_embed, len(val), axis=0)
        black_preds, _, _, _ = predict(net, x, black_embeds, device, args.batch_size)
        black_change = {"changed_count": int((preds != black_preds).sum()),
                        "changed_fraction": round(float((preds != black_preds).mean()), 6),
                        "black_macro_f1": overall(y, black_preds)["macro_f1"]}
        print("[BLACK-IMAGE]", black_change)

    output = {"rows": int(len(val)), "heads": model_scores,
              "image_gate_mean": float(gates.mean()), "black_image": black_change}
    (out / "summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print(f"\n[SAVED] Aggregate-only reports: {out}")
    print("[NOTE] These are diagnostic aggregate results; gate weights are not causal attributions.")


if __name__ == "__main__":
    main()
