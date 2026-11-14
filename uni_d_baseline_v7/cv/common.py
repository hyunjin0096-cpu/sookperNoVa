"""Common data loading and group CV tools. Never loads Test for training/CV."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

LABELS = list(range(1, 7))
CONDITIONS = ["tr_method", "bat_class", "bat_cc", "soc_status"]
REQUIRED = ["sample_id", "experiment_id", "relative_time_s", "label", *CONDITIONS]


def load_labeled(train_csv: str | Path, validation_csv: str | Path) -> pd.DataFrame:
    parts = []
    for source, path in [("train", train_csv), ("validation", validation_csv)]:
        df = pd.read_csv(path)
        missing = sorted(set(REQUIRED) - set(df.columns))
        if missing:
            raise ValueError(f"{source}: missing required columns: {missing}")
        if df["sample_id"].isna().any() or df["experiment_id"].isna().any():
            raise ValueError(f"{source}: null sample_id / experiment_id")
        if df["label"].isna().any() or not df["label"].isin(LABELS).all():
            raise ValueError(f"{source}: labels must be integers 1..6")
        if df.duplicated(["experiment_id", "relative_time_s"]).any():
            raise ValueError(f"{source}: duplicate experiment/timestamp")
        for col in CONDITIONS:
            nvalues = df.groupby("experiment_id")[col].nunique(dropna=False)
            if (nvalues != 1).any():
                raise ValueError(f"{source}: {col} is not constant within an experiment")
        df = df.copy()
        df["source_split"] = source
        parts.append(df)
    if set(parts[0].experiment_id) & set(parts[1].experiment_id):
        raise ValueError("Train and Validation contain shared experiments")
    all_df = pd.concat(parts, ignore_index=True)
    if all_df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id across labeled data")
    return all_df


def fold_assignment(df: pd.DataFrame, *, n_splits: int, seed: int) -> pd.Series:
    groups = df["experiment_id"].to_numpy()
    y = df["label"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignment = np.full(len(df), -1, dtype=int)
    for fold, (_, valid_idx) in enumerate(cv.split(np.zeros(len(df)), y, groups)):
        assignment[valid_idx] = fold
    if (assignment < 0).any():
        raise AssertionError("Unassigned rows")
    per_group = pd.DataFrame({"group": groups, "fold": assignment}).groupby("group").fold.nunique()
    if not (per_group == 1).all():
        raise AssertionError("Group leakage in split")
    return pd.Series(assignment, index=df.index, name="fold")


def assignment_score(df: pd.DataFrame, folds: pd.Series) -> dict[str, float]:
    """Lower is better. Audits distributions ONLY, never model scores.

    Weighted score is an engineering heuristic, not a statistical confidence bound.
    Class distribution uses sample-weighting; conditions use experiment-weighting.
    """
    temp = df[["experiment_id", "label", *CONDITIONS]].copy()
    temp["fold"] = np.asarray(folds)
    n = temp.fold.nunique()
    class_global = temp.label.value_counts(normalize=True).reindex(LABELS, fill_value=0).to_numpy()
    class_costs = []
    missing_class = 0
    for _, part in temp.groupby("fold"):
        p = part.label.value_counts(normalize=True).reindex(LABELS, fill_value=0).to_numpy()
        class_costs.append(np.abs(p - class_global).sum())
        missing_class += sum((part.label.value_counts().reindex(LABELS, fill_value=0) == 0))
    class_imbalance = float(np.mean(class_costs))
    groups = temp.drop_duplicates("experiment_id")
    cond_costs = {}
    for col in CONDITIONS:
        global_dist = groups[col].value_counts(normalize=True, dropna=False)
        costs = []
        for _, part in groups.groupby("fold"):
            dist = part[col].value_counts(normalize=True, dropna=False)
            costs.append(float((dist.reindex(global_dist.index, fill_value=0) - global_dist).abs().sum()))
        cond_costs[col] = float(np.mean(costs))
    row_counts = temp.groupby("fold").size().to_numpy()
    group_counts = groups.groupby("fold").size().to_numpy()
    row_imbalance = float(np.std(row_counts) / np.mean(row_counts))
    group_imbalance = float(np.std(group_counts) / np.mean(group_counts))
    score = (3.0 * class_imbalance + 1.5 * cond_costs["tr_method"]
             + 0.8 * cond_costs["bat_class"] + 0.5 * cond_costs["bat_cc"]
             + 0.5 * cond_costs["soc_status"] + 0.8 * row_imbalance
             + 0.3 * group_imbalance + 25.0 * missing_class)
    return {"score": float(score), "class_l1": class_imbalance,
            "missing_class_fold_pairs": int(missing_class),
            "rows_cv": row_imbalance, "groups_cv": group_imbalance,
            **{f"condition_l1_{col}": value for col, value in cond_costs.items()}}


def attach_folds(df: pd.DataFrame, fold_csv: str | Path) -> pd.DataFrame:
    fmap = pd.read_csv(fold_csv)
    if list(fmap.columns) != ["experiment_id", "fold"]:
        raise ValueError("fold CSV must contain exactly experiment_id,fold in this order")
    if fmap.experiment_id.isna().any() or fmap.experiment_id.duplicated().any():
        raise ValueError("fold CSV has missing/duplicate experiment_id")
    if fmap.fold.isna().any() or not np.all(fmap.fold == fmap.fold.astype(int)):
        raise ValueError("fold must be integer")
    if set(fmap.experiment_id) != set(df.experiment_id):
        raise ValueError("fold experiment IDs do not exactly match labeled data")
    assigned = df.merge(fmap, on="experiment_id", how="left", validate="many_to_one", sort=False)
    if assigned.fold.isna().any():
        raise AssertionError("missing fold rows")
    return assigned
