"""Grade OOF predictions. Pooled original-5s OOF is PRIMARY; fold/subgroup are support.

Input CSV: sample_id,pred_label,fold[,prob_1,...,prob_6].
Never use predictions from models trained on the same experiment.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
from cv.common import load_labeled, attach_folds, LABELS


def evaluate(ref: pd.DataFrame, preds: pd.DataFrame,
             allow_partial: bool = False):
    required = {"sample_id", "pred_label", "fold"}
    if not required.issubset(preds.columns):
        raise ValueError(f"Prediction CSV needs {required}")
    if preds.sample_id.duplicated().any():
        raise ValueError("OOF contains duplicate sample IDs")
    if not set(preds.sample_id).issubset(set(ref.sample_id)):
        raise ValueError("OOF has unknown sample_id")
    if not allow_partial and set(preds.sample_id) != set(ref.sample_id):
        raise ValueError(f"Incomplete OOF: {len(preds)} / {len(ref)}; pass --allow-partial for screening")
    joined = preds.merge(ref[["sample_id", "label", "fold", "tr_method", "bat_class"]],
                         on="sample_id", suffixes=("_pred", "_actual"), validate="one_to_one")
    if not np.array_equal(joined.fold_pred.astype(int).to_numpy(), joined.fold_actual.astype(int).to_numpy()):
        raise ValueError("OOF fold does not match the frozen experiment fold")
    if joined.pred_label.isna().any() or not joined.pred_label.isin(LABELS).all():
        raise ValueError("pred_label must be 1..6")
    if allow_partial:
        for fold in sorted(joined.fold_actual.unique()):
            exp_rows = ref[ref.fold == fold]
            pred_rows = joined[joined.fold_actual == fold]
            if set(exp_rows.sample_id) != set(pred_rows.sample_id):
                raise ValueError(f"Partial Fold {fold}; screening must cover COMPLETE selected folds")
    y = joined.label.astype(int)
    yp = joined.pred_label.astype(int)
    pooled = float(f1_score(y, yp, average="macro", labels=LABELS, zero_division=0))
    f_by_fold = []
    for f, part in joined.groupby("fold_actual"):
        f_by_fold.append({"fold": int(f), "rows": int(len(part)),
                          "macro_f1": float(f1_score(part.label, part.pred_label,
                                                     average="macro", labels=LABELS, zero_division=0))})
    by_fold = pd.DataFrame(f_by_fold).sort_values("fold")
    prec, rec, f1, support = precision_recall_fscore_support(y, yp, labels=LABELS, zero_division=0)
    by_class = pd.DataFrame({"label": LABELS, "precision": prec, "recall": rec,
                             "f1": f1, "support": support})
    by_group = []
    for col in ["tr_method", "bat_class"]:
        for name, part in joined.groupby(col):
            by_group.append({"subgroup": col, "value": name, "rows": len(part),
                             "macro_f1": f1_score(part.label, part.pred_label,
                                                  average="macro", labels=LABELS, zero_division=0)})
    confusion = pd.DataFrame(confusion_matrix(y, yp, labels=LABELS),
                             index=[f"true_{i}" for i in LABELS],
                             columns=[f"pred_{i}" for i in LABELS])
    return {"pooled_oof_macro_f1": pooled,
            "evaluated_rows": len(joined),
            "full_oof": len(joined) == len(ref),
            "fold_f1_mean": float(by_fold.macro_f1.mean()),
            "fold_f1_std_sample": float(by_fold.macro_f1.std(ddof=1)) if len(by_fold) > 1 else None,
            "fold_f1_min": float(by_fold.macro_f1.min())}, by_fold, by_class, pd.DataFrame(by_group), confusion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--validation-csv", required=True)
    p.add_argument("--folds", default="artifacts/cv/folds_exp.csv")
    p.add_argument("--oof", required=True)
    p.add_argument("--out-dir", default="artifacts/cv/evaluation")
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    ref = attach_folds(load_labeled(args.train_csv, args.validation_csv), args.folds)
    scores, folds, classes, subgroups, cm = evaluate(ref, pd.read_csv(args.oof), args.allow_partial)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    folds.to_csv(out / "fold_metrics.csv", index=False)
    classes.to_csv(out / "class_metrics.csv", index=False)
    subgroups.to_csv(out / "subgroup_metrics.csv", index=False)
    cm.to_csv(out / "confusion_matrix.csv")
    print("=== OOF EVALUATION ===")
    print(json.dumps(scores, indent=2, ensure_ascii=False))
    if not scores["full_oof"]:
        print("SCREENING ONLY. Partial OOF is NOT a definitive model-selection score.")
    print("Class metrics:\n", classes.to_string(index=False))
    print("Saved to:", out.resolve())


if __name__ == "__main__":
    main()
