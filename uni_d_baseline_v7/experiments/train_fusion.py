"""Train Uni-DTHON ResNet18+MLP Fusion on frozen group CV or all labeled data.

Only uses train and validation CSVs with 97 labeled experiments; NEVER reads Test.
CV evaluation uses a fixed epoch count; NO outer-fold early stopping.

Example smoke: python -m experiments.train_fusion --data-dir "$D" --mode cv --fold-ids 0 --epochs 1 --smoke
Example full:  python -m experiments.train_fusion --data-dir "$D" --mode cv --fold-ids 0 --epochs 15
Example final: python -m experiments.train_fusion --data-dir "$D" --mode final --epochs 15
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from cv.common import attach_folds, load_labeled
from features.sensor_preprocess import SensorPreprocessor
from features.image_transform import ThermalImageTransform
from features.multimodal_dataset import ImageSensorDataset
from fusion_model import FusionNet


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_loader(frame: pd.DataFrame, prep: SensorPreprocessor, transform: ThermalImageTransform,
                 batch_size: int, num_workers: int, train: bool, seed: int, device: torch.device):
    ds = ImageSensorDataset(frame, prep, image_transform=transform)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=device.type == "cuda", generator=gen,
                      persistent_workers=(num_workers > 0))


def train_epoch(model, loader, optimizer, criterion, scaler, device, amp, limit_batches=None):
    model.train()
    running, seen = 0.0, 0
    for n, batch in enumerate(loader):
        if limit_batches is not None and n >= limit_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        s = batch["sensor"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=amp):
            logits = model(x, s)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.detach().item()) * len(y)
        seen += len(y)
    return running / max(1, seen), seen


@torch.inference_mode()
def evaluate(model, loader, device, amp, limit_batches=None):
    model.eval()
    all_probs, all_labels, all_ids = [], [], []
    for n, batch in enumerate(loader):
        if limit_batches is not None and n >= limit_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        s = batch["sensor"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp):
            logits = model(x, s)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["label"].numpy())
        raw_ids = batch["sample_id"]
        all_ids.extend(raw_ids.tolist() if torch.is_tensor(raw_ids) else list(raw_ids))
    if not all_probs:
        raise ValueError("No validation batches were processed")
    return np.concatenate(all_probs), np.concatenate(all_labels), all_ids


def load_data(data_dir: Path) -> pd.DataFrame:
    data = load_labeled(data_dir / "train" / "train.csv",
                        data_dir / "validation" / "validation.csv")
    roots = {"train": data_dir / "train", "validation": data_dir / "validation"}
    data["_image_root"] = data["source_split"].map({k: str(v) for k, v in roots.items()})
    if data["_image_root"].isna().any():
        raise ValueError("Missing image root for labeled row")
    return data


def assert_new_artifacts(directory: Path):
    for target in (directory / "model.pt", directory / "sensor_preprocess.json", directory / "config.json"):
        if target.exists():
            raise FileExistsError(f"Existing trained model must be preserved: {target}. "
                                  "Use a new --out-dir or pass --overwrite explicitly.")


def train_one(train_df: pd.DataFrame, valid_df: pd.DataFrame | None, out: Path,
              args, fold: int | None, device: torch.device):
    if not args.smoke and not args.overwrite:
        assert_new_artifacts(out)
    seed_all(args.seed + (fold if fold is not None else 0))
    transform = ThermalImageTransform(size=args.image_size, mode=args.image_mode)
    prep = SensorPreprocessor().fit(train_df)
    train_loader = build_loader(train_df, prep, transform, args.batch_size, args.num_workers,
                                True, args.seed, device)
    valid_loader = None
    if valid_df is not None:
        valid_loader = build_loader(valid_df, prep, transform, args.batch_size,
                                    args.num_workers, False, args.seed, device)
    model = FusionNet(sensor_dim=len(prep.feature_columns),
                      pretrained_backbone=args.pretrained_backbone).to(device)
    criterion_weights = None
    if args.weighted_ce:
        counts = np.bincount(train_df["label"].to_numpy(dtype=int) - 1, minlength=6)
        # Smoothed inverse prevalence; the only labels inspected are OUTER TRAIN.
        weights = 1.0 / np.maximum(1.0, counts.astype(np.float64))
        weights = weights / weights.mean()
        criterion_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=criterion_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW([
        {"params": list(model.image_backbone.parameters()), "lr": args.image_lr},
        {"params": list(model.image_project.parameters()) + list(model.sensor_encoder.parameters())
         + list(model.classifier.parameters()), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    amp = (device.type == "cuda" and not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    print(f"device={device}, amp={amp}, image_size={args.image_size}, "
          f"sensor_D={len(prep.feature_columns)}, train={len(train_df)}, "
          f"valid={len(valid_df) if valid_df is not None else 0}", flush=True)
    started = time.time()
    for epoch in range(args.epochs):
        loss, seen = train_epoch(model, train_loader, optimizer, criterion, scaler, device, amp,
                                 limit_batches=2 if args.smoke else None)
        print(f"epoch {epoch + 1:02d}/{args.epochs}: train_loss={loss:.5f}, "
              f"observations={seen}, elapsed_s={time.time()-started:.0f}", flush=True)
    if args.smoke:
        if valid_loader is not None:
            probs, labels, _ = evaluate(model, valid_loader, device, amp, limit_batches=1)
            macro = f1_score(labels + 1, probs.argmax(axis=1) + 1,
                             average="macro", labels=[1, 2, 3, 4, 5, 6], zero_division=0)
            print(f"SMOKE ONLY (NOT VALID CV): {len(labels)} rows, F1={macro:.4f}", flush=True)
        print("FUSION TRAIN SMOKE OK — no model files saved", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "format": "unid-fusion-v1.1", "sensor_dim": len(prep.feature_columns),
        "image_transform": transform.config(), "labels": [1, 2, 3, 4, 5, 6],
        "epochs": args.epochs, "seed": args.seed, "fold": fold,
        "pretrained_backbone_training": bool(args.pretrained_backbone),
        "class_weighted_ce": bool(args.weighted_ce),
        "label_smoothing": float(args.label_smoothing),
        "inference_batch_size": 1,
    }
    torch.save(model.cpu().state_dict(), out / "model.pt")
    prep.save(out / "sensor_preprocess.json")
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SAVED", out / "model.pt", flush=True)
    if valid_loader is not None:
        # Evaluate using the same loader and model, no parameter fitting on validation.
        model.to(device)
        prob, label0, ids = evaluate(model, valid_loader, device, amp)
        if len(ids) != len(valid_df) or list(ids) != list(valid_df.sample_id):
            raise RuntimeError("Validation predictions are not aligned to original input rows")
        pred = prob.argmax(axis=1) + 1
        score = f1_score(label0 + 1, pred, average="macro", labels=[1, 2, 3, 4, 5, 6], zero_division=0)
        res = pd.DataFrame({"sample_id": ids, "fold": fold, "pred_label": pred})
        for j in range(6):
            res[f"prob_{j + 1}"] = prob[:, j]
        res.to_csv(out / "oof.csv", index=False)
        print(f"Fold {fold} OOF Macro F1={score:.5f}, rows={len(valid_df)}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--mode", choices=["cv", "final"], default="cv")
    p.add_argument("--fold-map", default="artifacts/cv/folds_exp.csv")
    p.add_argument("--fold-ids", type=int, nargs="+", default=[0])
    p.add_argument("--out-dir", default=None)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=160)
    p.add_argument("--image-mode", choices=["pad", "stretch"], default="pad")
    p.add_argument("--image-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--weighted-ce", action="store_true")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--pretrained-backbone", action="store_true",
                   help="Allow torchvision PUBLIC ImageNet weights during training; may download")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--smoke", action="store_true", help="two train batches/epoch; NO saved checkpoint")
    p.add_argument("--overwrite", action="store_true", help="Explicitly replace existing artifacts")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        p.error("epochs and batch-size must be positive; num-workers >= 0")
    if not 0 <= args.label_smoothing < 1:
        p.error("label-smoothing must lie in [0,1)")
    data_dir = Path(args.data_dir).expanduser().resolve()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    seed_all(args.seed)
    data = load_data(data_dir)
    out_root = Path(args.out_dir or ("artifacts/fusion_cv" if args.mode == "cv" else "weights"))
    if args.mode == "final":
        train_one(data.reset_index(drop=True), None, out_root, args, None, device)
    else:
        data = attach_folds(data, args.fold_map)
        if len(set(args.fold_ids)) != len(args.fold_ids) or not set(args.fold_ids).issubset(set(data.fold.unique())):
            raise ValueError("--fold-ids must be unique values from the frozen fold map")
        for fold in args.fold_ids:
            train = data[data.fold != fold].reset_index(drop=True)
            valid = data[data.fold == fold].reset_index(drop=True)
            train_one(train, valid, out_root / f"fold_{fold}", args, fold, device)
    print("FINISHED", flush=True)


if __name__ == "__main__":
    main()
