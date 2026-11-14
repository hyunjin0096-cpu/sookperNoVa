"""Train a frozen ConvNeXt-Tiny + sensor MLP + sample-wise gated fusion model.

Fivefold GroupKFold is train-only. Validation is an independent external check.
Backbone embeddings are computed once, but final checkpoint contains backbone weights;
test.py reads fresh images rather than caches or prebuilt test features.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import (
    GatedFusion, SensorNormalizer, augment_cadences, build_sensor_features,
    encode_images, image_paths, load_pretrained_backbone, make_backbone,
    set_determinism,
)


def macro_f1(y, pred):
    return float(f1_score(y, pred, labels=[1, 2, 3, 4, 5, 6], average="macro", zero_division=0))


def class_weights(labels):
    count = np.bincount(labels, minlength=6).astype(float)
    count = np.maximum(count, 1.0)
    weights = np.sqrt(np.median(count) / count)
    return np.clip(weights, 0.4, 3.0).astype(np.float32)


def predict_arrays(model, sensors, images, device, batch_size=2048):
    model.eval()
    all_preds, all_sensor, all_image, gates = [], [], [], []
    with torch.inference_mode():
        for lo in range(0, len(sensors), batch_size):
            hi = lo + batch_size
            x = torch.from_numpy(sensors[lo:hi]).to(device)
            im = torch.from_numpy(images[lo:hi]).to(device)
            a, b, c, g = model(x, im)
            all_preds.append(a.argmax(1).cpu().numpy() + 1)
            all_sensor.append(b.argmax(1).cpu().numpy() + 1)
            all_image.append(c.argmax(1).cpu().numpy() + 1)
            gates.append(g.mean(1).cpu().numpy())
    return (np.concatenate(all_preds), np.concatenate(all_sensor),
            np.concatenate(all_image), np.concatenate(gates))


def train_one(seed, train_x, train_img, train_y, eval_x, eval_img, eval_y,
              args, epochs, patience):
    set_determinism(seed)
    device = torch.device(args.device)
    net = GatedFusion(train_x.shape[1], dropout=args.dropout).to(device)
    loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights(train_y), device=device))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_img),
                            torch.from_numpy(train_y.astype(np.int64)))
    # The fusion head works on fixed embeddings. No train-time image transforms/caches in test.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    best_score = -1.0
    best_epoch = epochs
    best_state = None
    no_gain = 0
    for epoch in range(1, epochs + 1):
        net.train()
        losses = []
        for x, im, y in loader:
            x, im, y = x.to(device), im.to(device), y.to(device)
            logits, sensor_logits, image_logits, _ = net(x, im)
            obj = (loss(logits, y) + args.aux_weight *
                   (loss(sensor_logits, y) + loss(image_logits, y)))
            opt.zero_grad(set_to_none=True)
            obj.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            losses.append(float(obj.item()))
        if eval_x is not None:
            pred, _, _, _ = predict_arrays(net, eval_x, eval_img, device)
            score = macro_f1(eval_y + 1, pred)
            print(f"    epoch={epoch:02d} loss={np.mean(losses):.4f} heldout_F1={score:.4f}", flush=True)
            if score > best_score + 1e-5:
                best_score, best_epoch = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                no_gain = 0
            else:
                no_gain += 1
                if no_gain >= patience:
                    break
        elif epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(f"    epoch={epoch:02d}/{epochs} loss={np.mean(losses):.4f}", flush=True)
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, int(best_epoch), float(best_score)


def evaluate(model, x, im, y, device):
    pred, spred, ipred, gates = predict_arrays(model, x, im, device)
    cm = confusion_matrix(y, pred, labels=[1, 2, 3, 4, 5, 6])
    return {
        "macro_f1": macro_f1(y, pred),
        "sensor_head_macro_f1": macro_f1(y, spred),
        "image_head_macro_f1": macro_f1(y, ipred),
        "accuracy": float(accuracy_score(y, pred)),
        "mean_image_gate_weight": float(gates.mean()),
        "class_report": classification_report(y, pred, labels=[1, 2, 3, 4, 5, 6],
                                              output_dict=True, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "stage_3_to_2": int(cm[2, 1]),
        "stage_5_to_4": int(cm[4, 3]),
        "stage_5_to_6": int(cm[4, 5]),
        "stage_6_to_5": int(cm[5, 4]),
    }


def run(args):
    set_determinism(args.seed)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print("[DEVICE]", device, flush=True)
    print("[DESIGN] frozen ConvNeXt ImageNet encoder + trainable gated fusion; causal features", flush=True)
    root = Path(args.data_root)
    tr = pd.read_csv(root / "train/train.csv", dtype={"sample_id": "string", "experiment_id": "string"})
    va = pd.read_csv(root / "validation/validation.csv", dtype={"sample_id": "string", "experiment_id": "string"})
    if set(tr["experiment_id"]) & set(va["experiment_id"]):
        raise ValueError("Train/Validation experiment_id overlap")
    y = tr["label"].to_numpy(dtype=int)
    yva = va["label"].to_numpy(dtype=int)
    if not set(np.unique(y)).issubset(set(range(1, 7))):
        raise ValueError("Labels must be integers 1..6")

    print("[1] Build per-row CURRENT/PAST features (no future, no image statistics)", flush=True)
    x_orig = build_sensor_features(tr)
    x_va = build_sensor_features(va)
    print("[2] Cadence augmentation: thin ORIGINAL train experiments before recomputing histories", flush=True)
    x_aug, source_row = augment_cadences(tr, enabled=not args.no_cadence_augmentation)
    groups = tr["experiment_id"].astype(str).to_numpy()
    groups_aug = groups[source_row]
    y_aug = y[source_row] - 1
    print(f"  originals={len(tr)} train_views={len(source_row)} validation={len(va)} features={x_orig.shape[1]}")

    print("[3] Load pretrained ConvNeXt-Tiny; weights MUST be explicit", flush=True)
    if args.random_backbone_smoke_only:
        print("[WARNING] RANDOM BACKBONE: smoke-test only; NOT a competitive trained model")
        backbone = make_backbone()
    else:
        backbone = load_pretrained_backbone(args.pretrained_weights, args.download_pretrained)

    print("[4] Encode original train/validation images once (fresh images; no test cache)", flush=True)
    img_tr = encode_images(backbone, image_paths(tr, root / "train"), device,
                           args.image_batch_size, args.workers, args.image_size)
    img_va = encode_images(backbone, image_paths(va, root / "validation"), device,
                           args.image_batch_size, args.workers, args.image_size)
    img_aug = img_tr[source_row]
    print("  embeddings", img_tr.shape, img_va.shape, flush=True)
    # Backbone frozen; moving it to CPU avoids unnecessary GPU memory use during MLP training.
    backbone = backbone.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fold_scores, fold_epochs = [], []
    if args.cv_folds >= 2:
        n_exps = len(set(groups))
        if args.cv_folds > n_exps:
            raise ValueError("cv_folds exceeds unique train experiments")
        for fold, (fit_idx, hold_idx) in enumerate(
            GroupKFold(n_splits=args.cv_folds).split(x_orig, y, groups=groups), 1
        ):
            fit_groups = set(groups[fit_idx])
            mask = np.isin(groups_aug, list(fit_groups))
            norm = SensorNormalizer().fit(x_aug.iloc[np.flatnonzero(mask)])
            fold_train = norm.transform(x_aug.iloc[np.flatnonzero(mask)])
            fold_hold = norm.transform(x_orig.iloc[hold_idx])
            print(f"[CV {fold}/{args.cv_folds}] fit={mask.sum()} heldout={len(hold_idx)} experiments={len(fit_groups)}", flush=True)
            net, best_epoch, _ = train_one(
                args.seed + fold, fold_train, img_aug[mask], y_aug[mask],
                fold_hold, img_tr[hold_idx], y[hold_idx] - 1,
                args, args.epochs, args.patience)
            fold_pred, _, _, _ = predict_arrays(net, fold_hold, img_tr[hold_idx], device)
            score = macro_f1(y[hold_idx], fold_pred)
            print(f"  [CV RESULT] Macro F1={score:.4f}, best epoch={best_epoch}", flush=True)
            fold_scores.append(score)
            fold_epochs.append(best_epoch)
    else:
        print("[WARNING] CV skipped: use only for technical tests; competition requires group CV", flush=True)

    final_epochs = (int(round(float(np.median(fold_epochs)))) if fold_epochs else args.epochs)
    final_epochs = max(1, final_epochs)
    print(f"[5] Final training all TRAIN experiments for {final_epochs} epochs, validation untouched", flush=True)
    norm = SensorNormalizer().fit(x_aug)
    final_net, _, _ = train_one(args.seed, norm.transform(x_aug), img_aug,
                                y_aug, None, None, None, args, final_epochs, args.patience)
    print("[6] External Validation evaluation (NOT used to choose hyperparameters/epochs)", flush=True)
    metrics = evaluate(final_net, norm.transform(x_va), img_va, yva, device)
    for field in ("macro_f1", "sensor_head_macro_f1", "image_head_macro_f1", "accuracy",
                  "mean_image_gate_weight", "stage_3_to_2", "stage_5_to_4"):
        print("  ", field, "=", round(metrics[field], 5), flush=True)
    print("[CONFUSION MATRIX] rows actual 1..6, columns predicted 1..6")
    print(np.array(metrics["confusion_matrix"]))
    # Method-B: first thin the observed rows of each validation experiment,
    # THEN recompute causal sensor features. This is not mere label subsampling.
    va_views, va_sources = augment_cadences(va, enabled=True)
    ordered_va = va.sort_values(["experiment_id", "relative_time_s", "sample_id"], kind="mergesort")
    c_va = ordered_va.groupby("experiment_id", sort=False).cumcount().to_numpy()
    n10 = int(np.sum(c_va % 2 == 0))
    n15 = int(np.sum(c_va % 3 == 0))
    cadence_scores = {}
    cursor = len(va)
    for cadence, count in (("10s", n10), ("15s", n15)):
        end = cursor + count
        sel = va_sources[cursor:end]
        x = norm.transform(va_views.iloc[cursor:end])
        pred, _, _, _ = predict_arrays(final_net, x, img_va[sel], device)
        cadence_scores[cadence] = {"macro_f1": macro_f1(yva[sel], pred), "n": int(count)}
        print(f"  Method-B {cadence} macro F1={cadence_scores[cadence]['macro_f1']:.4f} n={count}")
        cursor = end
    assert cursor == len(va_views) and cursor == len(va) + n10 + n15
    method_scores = {}
    for method in sorted(va.tr_method.dropna().unique()):
        mask = va.tr_method.to_numpy() == method
        s, _, _, _ = predict_arrays(final_net, norm.transform(x_va)[mask], img_va[mask], device)
        method_scores[str(method)] = {"macro_f1": macro_f1(yva[mask], s), "n": int(mask.sum())}
        print(f"  tr_method={method} Macro F1={method_scores[str(method)]['macro_f1']:.4f} N={mask.sum()}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Self-contained inference: backbone weights + head + immutable train-only normalization.
    checkpoint = {
        "model_type": "uni_dthon_convnext_tiny_gated_v1",
        "backbone_state": {k: v.detach().cpu() for k, v in backbone.state_dict().items()},
        "fusion_state": {k: v.detach().cpu() for k, v in final_net.cpu().state_dict().items()},
        "normalizer": norm.to_dict(),
        "sensor_dim": x_orig.shape[1], "image_size": args.image_size,
        "dropout": args.dropout, "pretrained": not args.random_backbone_smoke_only,
        "classes": [1, 2, 3, 4, 5, 6],
        "feature_spec": "sensor_causal_v1_10_30_60_120s",
        "train_augmentation": "original+10s+15s" if not args.no_cadence_augmentation else "original",
    }
    torch.save(checkpoint, output)
    score_report = {
        "cv_fold_macro_f1": fold_scores,
        "cv_macro_f1_mean": float(np.mean(fold_scores)) if fold_scores else None,
        "cv_macro_f1_std": float(np.std(fold_scores)) if fold_scores else None,
        "cv_best_epochs": fold_epochs,
        "final_train_epochs": final_epochs,
        "train_original_n": len(tr),
        "train_augmented_n": len(source_row),
        "validation_n": len(va),
        "validation": metrics,
        "validation_method_b": cadence_scores,
        "validation_by_tr_method": method_scores,
        "source": "EDA-driven hypotheses; no Test labels/fit/whole-experiment stats",
    }
    report_path = output.parent / "validation_report.json"
    report_path.write_text(json.dumps(score_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[SAVED MODEL]", output, "size MiB", round(output.stat().st_size / 1024**2, 2))
    print("[SAVED AGGREGATE REPORT]", report_path)
    print("[NEXT] python test.py --input /path/to/test.csv --image_root /path/to/test --output submission.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("weights/model.pt"))
    ap.add_argument("--pretrained-weights", type=Path, help="Local torchvision ConvNeXt-Tiny ImageNet state_dict")
    ap.add_argument("--download-pretrained", action="store_true", help="Only if contest rules permit downloads")
    ap.add_argument("--random-backbone-smoke-only", action="store_true", help="DEBUG ONLY. Not a real experiment!")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--image-batch-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--no-cadence-augmentation", action="store_true")
    ap.add_argument("--aux-weight", type=float, default=0.20)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--lr", type=float, default=0.0008)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    run(ap.parse_args())
