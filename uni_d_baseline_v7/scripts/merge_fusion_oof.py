"""Combine saved per-fold oof.csv outputs from the frozen 5-fold Fusion runs."""
import argparse
from pathlib import Path
import pandas as pd

from cv.common import load_labeled, attach_folds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--fold-map", default="artifacts/cv/folds_exp.csv")
    p.add_argument("--model-dir", default="artifacts/fusion_cv")
    p.add_argument("--output", default="artifacts/fusion_cv/fusion_oof.csv")
    args = p.parse_args()
    d = Path(args.data_dir)
    reference = attach_folds(load_labeled(d / "train/train.csv", d / "validation/validation.csv"), args.fold_map)
    parts = []
    for fold in sorted(reference.fold.unique()):
        f = Path(args.model_dir) / f"fold_{int(fold)}" / "oof.csv"
        if not f.is_file():
            raise FileNotFoundError(f"Fold {fold} was not trained: {f}")
        part = pd.read_csv(f)
        if not part.fold.eq(fold).all():
            raise ValueError(f"Wrong fold column inside {f}")
        parts.append(part)
    joined = pd.concat(parts, ignore_index=True)
    if joined.sample_id.duplicated().any() or set(joined.sample_id) != set(reference.sample_id):
        raise ValueError("Missing or duplicate sample IDs in merged OOF")
    correct = joined.merge(reference[["sample_id", "fold"]], on="sample_id", suffixes=("_pred", "_ref"))
    if not correct.fold_pred.eq(correct.fold_ref).all():
        raise ValueError("Predictions do not match frozen fold IDs")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out, index=False)
    print("Saved validated full OOF:", out, "rows", len(joined))


if __name__ == "__main__":
    main()
