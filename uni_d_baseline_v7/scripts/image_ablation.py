"""Image ablation on frozen Uni-DTHON OOF Fusion checkpoints.

Evaluates each validation fold with its OWN trained checkpoint and saved
sensor preprocessor. Normal image vs pure black RGB (same normalized pipeline).
No training, no refitting, no Test data, no checkpoint modifications.

Usage:
 python -m scripts.image_ablation --data-dir "$D" \
   --folds artifacts/cv/folds_exp.csv \
   --model-dir artifacts/fusion_cv \
   --fusion-oof artifacts/fusion_cv/fusion_oof.csv \
   --sensor-oof artifacts/cv/sensor_example_oof.csv \
   --out-dir artifacts/image_ablation
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from cv.common import attach_folds
from experiments.train_fusion import load_data
from features.sensor_preprocess import SensorPreprocessor
from features.image_transform import ThermalImageTransform
from features.multimodal_dataset import ImageSensorDataset
from fusion_model import FusionNet

LABELS = [1, 2, 3, 4, 5, 6]
PROBS = [f"prob_{i}" for i in LABELS]


def get_model_and_preprocessor(model_dir: Path, device: torch.device):
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("format") != "unid-fusion-v1.1" or config.get("labels") != LABELS:
        raise ValueError(f"Unexpected model config in {model_dir}")
    p = config["image_transform"]
    tfm = ThermalImageTransform(size=int(p["size"]), mode=p["mode"],
                                mean=tuple(p["mean"]), std=tuple(p["std"]),
                                pad_rgb=tuple(p["pad_rgb"]))
    prep = SensorPreprocessor.load(model_dir / "sensor_preprocess.json")
    if len(prep.feature_columns) != int(config["sensor_dim"]):
        raise ValueError("Saved sensor feature dimensions do not match checkpoint")
    model = FusionNet(sensor_dim=int(config["sensor_dim"]),
                      pretrained_backbone=False)
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, prep, tfm


@torch.inference_mode()
def infer_pair(model, loader, tfm, device, use_amp: bool):
    # Same RGB image as organizer blank-image replacement, then normalization.
    blank_pil = Image.new("RGB", (tfm.size, tfm.size), (0, 0, 0))
    blank_img = torch.from_numpy(tfm(blank_pil)).unsqueeze(0).to(device)
    all_normal, all_blank, all_ids, all_labels = [], [], [], []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        sensor = batch["sensor"].to(device, non_blocking=True)
        blank = blank_img.expand(img.shape[0], -1, -1, -1)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            normal_logits = model(img, sensor)
            blank_logits = model(blank, sensor)
        all_normal.append(torch.softmax(normal_logits.float(), dim=1).cpu().numpy())
        all_blank.append(torch.softmax(blank_logits.float(), dim=1).cpu().numpy())
        all_labels.append(batch["label"].numpy() + 1)
        ids = batch["sample_id"]
        all_ids.extend(ids.tolist() if torch.is_tensor(ids) else list(ids))
    if not all_normal:
        raise ValueError("No samples were evaluated")
    return (np.concatenate(all_normal), np.concatenate(all_blank),
            np.concatenate(all_labels), all_ids)


def f1(y, yhat):
    return float(f1_score(y, yhat, average="macro", labels=LABELS,
                          zero_division=0))


def metrics(y, normal, blank):
    a = normal.argmax(axis=1) + 1
    b = blank.argmax(axis=1) + 1
    diff = np.abs(normal - blank)
    return {
        "rows": int(len(y)),
        "normal_macro_f1": f1(y, a),
        "blank_macro_f1": f1(y, b),
        "macro_f1_drop_normal_minus_blank": f1(y, a) - f1(y, b),
        "changed_label_count": int((a != b).sum()),
        "changed_label_rate": float((a != b).mean()),
        "mean_abs_probability_change": float(diff.mean()),
        "mean_total_variation": float((diff.sum(axis=1) / 2).mean()),
        "max_abs_probability_change": float(diff.max()),
    }


def checked_prob_table(path: Path, ids: pd.Series, prefix: str) -> np.ndarray:
    table = pd.read_csv(path)
    need = {"sample_id", *PROBS}
    if not need.issubset(table.columns):
        raise ValueError(f"Missing expected columns in {path}: {need - set(table.columns)}")
    if table.sample_id.isna().any() or table.sample_id.duplicated().any():
        raise ValueError(f"Missing/duplicate sample_id in {path}")
    reference = pd.DataFrame({"sample_id": ids.tolist()})
    merged = reference.merge(table[["sample_id"] + PROBS], on="sample_id",
                             validate="one_to_one", sort=False, indicator=True)
    if (merged._merge != "both").any() or len(merged) != len(reference):
        raise ValueError(f"Missing {prefix} OOF sample IDs; don't compare partial results")
    p = merged[PROBS].to_numpy(dtype=np.float64)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError(f"Invalid {prefix} probabilities")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError(f"{prefix} probability rows do not sum to 1")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--folds", default="artifacts/cv/folds_exp.csv")
    ap.add_argument("--model-dir", default="artifacts/fusion_cv")
    ap.add_argument("--fusion-oof", default="artifacts/fusion_cv/fusion_oof.csv")
    ap.add_argument("--sensor-oof", default=None,
                    help="Optional sensor OOF for 50:50 ensemble normal vs blank")
    ap.add_argument("--out-dir", default="artifacts/image_ablation")
    ap.add_argument("--fold-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Use 64 to match original Fusion OOF validation batch size")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--fp32", action="store_true", help="Disable AMP, will differ slightly from stored OOF")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or len(set(args.fold_ids)) != len(args.fold_ids):
        ap.error("Invalid batch-size, num-workers or duplicate fold IDs")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA requested but not available")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    torch.manual_seed(2026)
    np.random.seed(2026)
    random.seed(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    use_amp = device.type == "cuda" and not args.fp32

    df = attach_folds(load_data(Path(args.data_dir).expanduser()), args.folds)
    if not set(args.fold_ids).issubset(set(df.fold.unique())):
        raise ValueError("fold IDs not present in frozen fold map")
    print(f"Image Ablation: device={device}, AMP={use_amp}, folds={args.fold_ids}, "
          f"batch={args.batch_size}", flush=True)
    outputs = []
    fold_metrics = []
    for fold in args.fold_ids:
        frame = df[df.fold == fold].reset_index(drop=True)
        model, prep, tfm = get_model_and_preprocessor(Path(args.model_dir) / f"fold_{fold}", device)
        ds = ImageSensorDataset(frame, prep, image_transform=tfm)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=device.type == "cuda",
                            persistent_workers=args.num_workers > 0)
        pn, pb, labels, ids = infer_pair(model, loader, tfm, device, use_amp)
        if len(frame) != len(ids) or list(frame.sample_id) != list(ids):
            raise AssertionError(f"Fold {fold}: predictions do not align with CSV")
        if not np.array_equal(labels, frame.label.to_numpy(dtype=int)):
            raise AssertionError("Unexpected label alignment")
        stat = {"fold": int(fold), **metrics(labels, pn, pb)}
        fold_metrics.append(stat)
        print(f"Fold {fold}: normal F1={stat['normal_macro_f1']:.5f}, "
              f"blank F1={stat['blank_macro_f1']:.5f}, "
              f"label changed={stat['changed_label_rate']:.2%}, "
              f"prob change={stat['mean_abs_probability_change']:.5f}", flush=True)
        block = pd.DataFrame({
            "sample_id": ids, "experiment_id": frame.experiment_id.to_numpy(),
            "fold": int(fold), "label": labels,
            "tr_method": frame.tr_method.to_numpy(),
            "pred_normal": pn.argmax(axis=1) + 1,
            "pred_blank": pb.argmax(axis=1) + 1,
            "image_changes_label": (pn.argmax(axis=1) != pb.argmax(axis=1)).astype(int),
            "mean_abs_probability_change": np.abs(pn - pb).mean(axis=1),
        })
        for i in range(6):
            block[f"normal_prob_{i + 1}"] = pn[:, i]
            block[f"blank_prob_{i + 1}"] = pb[:, i]
        outputs.append(block)
        del model, loader, ds
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = pd.concat(outputs, ignore_index=True)
    if out.sample_id.duplicated().any():
        raise AssertionError("Duplicated sample IDs across evaluated folds")
    expected_rows = int(df.fold.isin(args.fold_ids).sum())
    if len(out) != expected_rows:
        raise AssertionError("Incomplete evaluated OOF rows")
    y = out.label.to_numpy(dtype=int)
    pn = out[[f"normal_prob_{i}" for i in LABELS]].to_numpy(dtype=np.float64)
    pb = out[[f"blank_prob_{i}" for i in LABELS]].to_numpy(dtype=np.float64)
    all_stats = metrics(y, pn, pb)
    print("\n=== FUSION IMAGE ABLATION (OOF) ===", flush=True)
    print(json.dumps(all_stats, indent=2), flush=True)

    summary = {"model": "Fusion v1.1", "n_folds_evaluated": len(args.fold_ids),
               "full_oof": bool(len(args.fold_ids) == df.fold.nunique() and len(out) == len(df)),
               "device": str(device), "amp": bool(use_amp),
               "fusion": all_stats}
    ref_file = Path(args.fusion_oof)
    if ref_file.is_file():
        ref = checked_prob_table(ref_file, out.sample_id, "Fusion")
        mismatch = float(((ref.argmax(axis=1) + 1) != out.pred_normal.to_numpy()).mean())
        delta = float(np.abs(ref - pn).mean())
        summary["recompute_vs_saved_oof"] = {
            "label_mismatch_rate": mismatch, "mean_abs_probability_difference": delta
        }
        print(f"\nSaved Fusion OOF vs recalculated normal: label mismatch={mismatch:.3%}, "
              f"mean |prob diff|={delta:.7f}", flush=True)
        if mismatch > 0.005:
            print("WARNING: >0.5% original OOF mismatch. Check batch-size/AMP and code versions!", flush=True)
    else:
        print(f"WARNING: reference Fusion OOF not found: {ref_file}", flush=True)

    if args.sensor_oof:
        ps = checked_prob_table(Path(args.sensor_oof), out.sample_id, "Sensor")
        ens_normal = (ps + pn) / 2
        ens_blank = (ps + pb) / 2
        ens_metrics = metrics(y, ens_normal, ens_blank)
        summary["sensor_oof_macro_f1"] = f1(y, ps.argmax(axis=1) + 1)
        summary["ensemble_5050"] = ens_metrics
        print("\n=== ENSEMBLE 50:50 IMAGE ABLATION ===", flush=True)
        print(f"Sensor-only OOF F1: {summary['sensor_oof_macro_f1']:.6f}", flush=True)
        print(json.dumps(ens_metrics, indent=2), flush=True)

    class_reports = []
    for condition, pred in [("normal", pn.argmax(axis=1) + 1),
                            ("blank", pb.argmax(axis=1) + 1)]:
        report = classification_report(y, pred, labels=LABELS, output_dict=True,
                                       zero_division=0)
        for label in LABELS:
            row = report[str(label)]
            class_reports.append({"condition": condition, "label": label,
                                  "precision": float(row["precision"]),
                                  "recall": float(row["recall"]),
                                  "f1": float(row["f1-score"]),
                                  "support": int(row["support"])})

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    out.to_csv(output / "image_ablation_oof.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(class_reports).to_csv(output / "class_metrics.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(f"\nSaved results to {output}", flush=True)
    print("Diagnostic only: probability differences do NOT prove physical-image usefulness.", flush=True)
    print("This is labeled OOF analysis, NOT a DAKER submission.", flush=True)


if __name__ == "__main__":
    main()
