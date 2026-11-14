"""Audit frozen folds for leakage, rare classes, and experiment conditions."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .common import load_labeled, attach_folds, assignment_score, LABELS, CONDITIONS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--validation-csv", required=True)
    p.add_argument("--folds", default="artifacts/cv/folds_exp.csv")
    p.add_argument("--out-dir", default="artifacts/cv/audit")
    args = p.parse_args()
    df = attach_folds(load_labeled(args.train_csv, args.validation_csv), args.folds)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nfold = df.fold.nunique()
    if list(sorted(df.fold.unique())) != list(range(nfold)):
        raise ValueError("Fold ids must be consecutive 0..K-1")
    if df.groupby("experiment_id").fold.nunique().max() != 1:
        raise AssertionError("Group leakage")
    if df.sample_id.duplicated().any():
        raise AssertionError("OOF sample duplicates")
    class_table = (pd.crosstab(df.fold, df.label).reindex(columns=LABELS, fill_value=0))
    class_table.to_csv(out / "fold_class_counts.csv")
    group_meta = df.drop_duplicates("experiment_id")
    fold_summary = df.groupby("fold").agg(rows=("sample_id", "size"),
                                          experiments=("experiment_id", "nunique"))
    fold_summary.to_csv(out / "fold_summary.csv")
    cond_tables = []
    for col in CONDITIONS:
        tab = pd.crosstab(group_meta.fold, group_meta[col])
        tab.to_csv(out / f"fold_{col}_groups.csv")
        cond_tables.append(f"{col}: {tab.to_dict()}")
    metrics = assignment_score(df, df.fold)
    with (out / "audit_summary.json").open("w", encoding="utf-8") as fp:
        json.dump({"folds": nfold, "rows": len(df), "experiments": df.experiment_id.nunique(),
                   "group_overlap": 0, "score_components": metrics,
                   "warning": "Audit is not a proof of feature causality; run separate inference selfchecks."},
                  fp, indent=2, ensure_ascii=False)
    print("=== FOLD AUDIT ===")
    print(fold_summary.to_string())
    print("\nClass rows in each fold (columns=labels 1..6):")
    print(class_table.to_string())
    print(f"\nAudit score (lower better): {metrics['score']:.4f}")
    print(f"Missing class/fold pairs: {metrics['missing_class_fold_pairs']}")
    if metrics["missing_class_fold_pairs"]:
        print("WARNING: at least one fold is missing a class!")
    print(f"Details: {out.resolve()} (per-condition counts by experiment)")


if __name__ == "__main__":
    main()
