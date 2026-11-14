"""v2_tune_stack.py - 2단계 LightGBM '규제 강도'를 비교 (CV 고정 fold, 모델 재학습은 GBDT만이라 가볍다)

배경: 실험은 97개뿐이고 한 실험이 약 450행이라, 잎 하나(min_data_in_leaf=60)가 '한 실험'만 담으면 실험 고유 패턴을 외운다.
      잎을 크게(여러 실험을 포함) 하고 트리를 작게 하면 새 실험에 대한 일반화가 좋아질 수 있다.
비교 대상: 센서만(B) vs 센서+이미지(C). 선택 기준은 풀링 OOF Macro F1 (fold 최소값도 같이 확인).

사용:  python -u -m v2_tune_stack --data-dir "$D" --folds artifacts/cv/folds_exp.csv 2>&1 | tee v2_tune.log
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from cv.common import attach_folds, load_labeled
from v2_common import LABELS, balanced_weights, build_feature_table

try:
    import lightgbm as lgb
except Exception:
    lgb = None

PCOLS = [f"prob_{k}" for k in LABELS]
CONFIGS = [  # (이름, 잎 수, min_data_in_leaf, 라운드, feature_fraction, lambda_l2, extra_trees)
    ("기준(31잎,60)", 31, 60, 300, 0.7, 5.0, False),
    ("15잎,200", 15, 200, 250, 0.7, 5.0, False),
    ("15잎,600", 15, 600, 250, 0.7, 10.0, False),
    ("7잎,300", 7, 300, 300, 0.5, 10.0, False),
    ("7잎,1000", 7, 1000, 300, 0.5, 20.0, False),
    ("15잎,300,extra", 15, 300, 300, 0.5, 10.0, True),
]


def f1m(y, p):
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def fit_predict(Xtr, ytr, Xva, cfg, seed=100):
    _, leaves, mdl, rounds, ff, l2, et = cfg
    w = balanced_weights(ytr)
    if lgb is not None:
        params = dict(objective="multiclass", num_class=6, learning_rate=0.05, num_leaves=leaves, min_data_in_leaf=mdl,
                      feature_fraction=ff, bagging_fraction=0.8, bagging_freq=1, lambda_l2=l2, extra_trees=et,
                      seed=seed, deterministic=True, force_row_wise=True, num_threads=4, verbosity=-1)
        m = lgb.train(params, lgb.Dataset(Xtr.to_numpy(float), label=ytr - 1, weight=w), num_boost_round=rounds)
        return np.asarray(m.predict(Xva.to_numpy(float))).reshape(len(Xva), 6)
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=rounds, max_leaf_nodes=leaves, min_samples_leaf=min(mdl, 200),
                                       learning_rate=0.1, early_stopping=False, random_state=seed)
    m.fit(Xtr.to_numpy(float), ytr, sample_weight=w)
    out = np.zeros((len(Xva), 6))
    for k, c in enumerate(m.classes_):
        out[:, int(c) - 1] = m.predict_proba(Xva.to_numpy(float))[:, k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--folds", default="artifacts/cv/folds_exp.csv")
    ap.add_argument("--img-oof", default="artifacts/v2/img_oof.csv")
    ap.add_argument("--out-dir", default="artifacts/v2")
    ap.add_argument("--variants", nargs="+", default=["B", "C"])
    ap.add_argument("--configs", type=int, nargs="+", default=None, help="CONFIGS 번호(0부터)만 실행")
    a = ap.parse_args()
    D = Path(a.data_dir).expanduser()
    df = attach_folds(load_labeled(str(D / "train/train.csv"), str(D / "validation/validation.csv")), a.folds).reset_index(drop=True)
    df["fold"] = df["fold"].astype(int)
    y, fold = df["label"].to_numpy(int), df["fold"].to_numpy(int)
    o = pd.read_csv(a.img_oof).set_index("sample_id")
    P = o.loc[df["sample_id"], PCOLS].to_numpy(float)
    X = {v: build_feature_table(df, P, v) for v in a.variants}
    cfgs = [CONFIGS[i] for i in a.configs] if a.configs else CONFIGS
    rows = []
    for cfg in cfgs:
        res = {"설정": cfg[0]}
        t0 = time.time()
        for v in a.variants:
            oof = np.zeros((len(df), 6))
            for k in sorted(set(fold)):
                tr, va = fold != k, fold == k
                oof[va] = fit_predict(X[v][tr], y[tr], X[v][va], cfg)
            pred = oof.argmax(1) + 1
            fm = [f1m(y[fold == k], pred[fold == k]) for k in sorted(set(fold))]
            res[f"{v} 풀링"], res[f"{v} fold최소"] = f1m(y, pred), min(fm)
            pd.DataFrame(oof, columns=PCOLS).assign(sample_id=df["sample_id"], fold=fold).to_csv(
                Path(a.out_dir) / f"tune_oof_{v}_{cfgs.index(cfg)}.csv", index=False)
        if "B" in a.variants and "C" in a.variants:
            res["이미지효과 C-B"] = res["C 풀링"] - res["B 풀링"]
        rows.append(res)
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()) +
              f"  ({time.time() - t0:.0f}s)", flush=True)
    print("\n=== 요약 ===")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("번호는 CONFIGS 순서(0부터). 가장 좋은 설정과 이미지 효과 열을 알려 주세요.")


if __name__ == "__main__":
    main()