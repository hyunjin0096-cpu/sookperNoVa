"""Runnable SENSOR-ONLY CV reference; NOT a valid final Uni-D submission.

Uses Group-held-out folds and causal features, training sklearn HistGradientBoosting.
Optional gap-view is a PAIRED robustness assessment, not the primary OOF score.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from cv.common import load_labeled, attach_folds, LABELS
from features.causal_sensor_features import make_causal_sensor_features
from features.gap_view import make_gap_view


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--validation-csv", required=True)
    p.add_argument("--folds", default="artifacts/cv/folds_exp.csv")
    p.add_argument("--out", default="artifacts/cv/sensor_example_oof.csv")
    p.add_argument("--fold-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="0 1 for quick screening, ALL 0..4 for official OOF")
    p.add_argument("--max-iter", type=int, default=75)
    p.add_argument("--gap-check", action="store_true")
    p.add_argument("--gap-seed", type=int, default=2026)
    args = p.parse_args()
    df = attach_folds(load_labeled(args.train_csv, args.validation_csv), args.folds)
    if len(set(args.fold_ids)) != len(args.fold_ids) or not set(args.fold_ids).issubset(set(df.fold.unique())):
        raise ValueError("fold-ids must be unique, valid fold indices")
    features = make_causal_sensor_features(df).drop(columns="sample_id")
    cols = list(features.columns)
    # Experimental sensor-only model uses no images. It must never be submitted alone.
    results, pair_results = [], []
    for fold in sorted(args.fold_ids):
        train_mask = df.fold != fold
        valid_mask = df.fold == fold
        tr_idx = np.flatnonzero(train_mask.to_numpy())
        va_idx = np.flatnonzero(valid_mask.to_numpy())
        model = HistGradientBoostingClassifier(
            max_iter=args.max_iter, max_leaf_nodes=31, learning_rate=0.07,
            class_weight="balanced", l2_regularization=1.0,
            early_stopping=False, random_state=42,
        )
        model.fit(features.iloc[tr_idx], df.label.iloc[tr_idx])
        probabilities = model.predict_proba(features.iloc[va_idx])
        all_probs = np.zeros((len(va_idx), 6), dtype=float)
        for k, cls in enumerate(model.classes_):
            all_probs[:, int(cls) - 1] = probabilities[:, k]
        chosen = np.array(LABELS)[all_probs.argmax(axis=1)]
        prediction = pd.DataFrame({"sample_id": df.sample_id.iloc[va_idx].to_numpy(),
                                   "fold": fold, "pred_label": chosen})
        for idx, cls in enumerate(LABELS):
            prediction[f"prob_{cls}"] = all_probs[:, idx]
        results.append(prediction)
        actual = df.iloc[va_idx].label
        f1 = f1_score(actual, chosen, average="macro", labels=LABELS, zero_division=0)
        print(f"Fold {fold}: rows={len(va_idx)}, Macro F1={f1:.4f}", flush=True)
        if args.gap_check:
            # This removes entire prior rows before recalculating features.
            full_valid = df.iloc[va_idx].copy().reset_index(drop=True)
            thinner = make_gap_view(full_valid, seed=args.gap_seed)
            thin_features = make_causal_sensor_features(thinner).drop(columns="sample_id")[cols]
            thin_probs = model.predict_proba(thin_features)
            thin_class = model.classes_[thin_probs.argmax(axis=1)]
            paired = thinner[["sample_id", "label"]].copy()
            paired["pred_thin"] = thin_class
            paired = paired.merge(prediction[["sample_id", "pred_label"]], on="sample_id", validate="one_to_one")
            paired["fold"] = fold
            pair_results.append(paired)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(results, ignore_index=True).to_csv(out, index=False)
    print("Saved OOF:", out.resolve())
    if args.gap_check:
        pair = pd.concat(pair_results, ignore_index=True)
        base = f1_score(pair.label, pair.pred_label, labels=LABELS, average="macro", zero_division=0)
        thin = f1_score(pair.label, pair.pred_thin, labels=LABELS, average="macro", zero_division=0)
        print(f"Gap paired assessment (same {len(pair)} retained rows): "
              f"full history F1={base:.4f}, thinned history F1={thin:.4f}")
        print("NOTE: this is a robustness diagnostic, not a leaderboard score estimate.")
        pair.to_csv(out.with_name(out.stem + "_paired_gap.csv"), index=False)
    print("WARNING: SENSOR ONLY. Final competition submission MUST genuinely use images too.")


if __name__ == "__main__":
    main()
