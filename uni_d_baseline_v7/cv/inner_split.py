"""Inner GROUP split for early stopping; outer valid fold must not tune epochs.

Typical use:
  outer_train = all_df[all_df.fold != fold].reset_index(drop=True)
  inner_train_pos, inner_val_pos = make_inner_split(outer_train, seed=2026)
  # fit preprocess ONLY on inner_train_pos; early stop on inner_val_pos;
  # record best_epoch. Then REINITIALIZE model and train on all outer_train
  # for best_epoch epochs; evaluate once on outer validation fold.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_inner_split(outer_train: pd.DataFrame, seed: int = 2026,
                     n_splits: int = 4) -> tuple[np.ndarray, np.ndarray]:
    needed = {"experiment_id", "label"}
    if not needed.issubset(outer_train):
        raise ValueError(f"Expected {needed}")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    inner_tr, inner_val = next(cv.split(np.zeros(len(outer_train)),
                                        outer_train.label, outer_train.experiment_id))
    assert not (set(outer_train.iloc[inner_tr].experiment_id)
                & set(outer_train.iloc[inner_val].experiment_id))
    return inner_tr, inner_val
