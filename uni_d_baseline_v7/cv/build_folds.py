"""Create frozen experiment-aware 5-fold assignment from labeled data."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .common import load_labeled, fold_assignment, assignment_score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--validation-csv", required=True)
    p.add_argument("--out-dir", default="artifacts/cv")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--force", action="store_true", help="Explicitly overwrite frozen folds")
    args = p.parse_args()
    if args.seeds < 1 or args.n_splits < 2:
        p.error("--seeds >= 1 and --n-splits >= 2 required")
    df = load_labeled(args.train_csv, args.validation_csv)
    out_dir = Path(args.out_dir)
    out = out_dir / "folds_exp.csv"
    if out.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite frozen folds: {out}. Use --force ONLY intentionally.")
    results = []
    best = None
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        folds = fold_assignment(df, n_splits=args.n_splits, seed=seed)
        metrics = assignment_score(df, folds)
        results.append({"seed": seed, **metrics})
        if best is None or metrics["score"] < best[0]:
            best = (metrics["score"], seed, folds, metrics)
    _, chosen_seed, assigned, metrics = best
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = pd.DataFrame({"experiment_id": df.experiment_id, "fold": assigned})
    mapping = mapping.drop_duplicates("experiment_id").sort_values("experiment_id")
    mapping.to_csv(out, index=False)
    pd.DataFrame(results).sort_values(["score", "seed"]).to_csv(out_dir / "seed_search.csv", index=False)
    with open(out_dir / "split_config.json", "w", encoding="utf-8") as fp:
        json.dump({"n_splits": args.n_splits, "chosen_seed": chosen_seed,
                   "seeds_tested": args.seeds, "audit_score": metrics,
                   "labeled_rows": int(len(df)),
                   "experiments": int(df.experiment_id.nunique()),
                   "selection_criterion": "labels and experiment conditions only; NO model performance"},
                  fp, indent=2, ensure_ascii=False)
    print(f"Saved: {out} | experiments={len(mapping)} | seed={chosen_seed} | audit score={metrics['score']:.4f}")
    print("Freeze this file. Use the SAME fold mapping for every candidate model.")


if __name__ == "__main__":
    main()
