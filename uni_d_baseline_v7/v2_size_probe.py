"""v2_size_probe.py - 이미지 파일의 '크기(가로x세로)'가 단계 예측에 도움이 되는지 점검하고, 도움이 되는지 CV로 비교

배경: EDA에서 이미지 크기/종횡비가 단계(또는 실험)와 상관이 있었다. 지금 이미지 모델은 128x128로 강제 리사이즈해서
      그 정보를 모두 버린다. 크기는 PNG 헤더만 읽으면 되므로 빠르고, test.py에서도 같은 방식으로 읽을 수 있다(과거 행까지만 사용).
출력: 1) 크기 분포와 단계별 크기  2) Test 이미지가 학습에서 본 크기와 같은지  3) CV 비교(센서 B vs 센서+크기)

사용:  python -u -m v2_size_probe --data-dir "$D" --folds artifacts/cv/folds_exp.csv 2>&1 | tee v2_size.log
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score

from cv.common import attach_folds, load_labeled
from v2_common import LABELS, build_feature_table, exp_groups, fit_gb, predict_gb, resolve_image

SIZE_COLS = ["sz_w", "sz_h", "sz_aspect", "sz_area", "sz_dw0", "sz_dh0", "sz_changed", "sz_nchg", "sz_since_chg_s",
             "sz_wmax", "sz_hmax", "sz_wmin", "sz_hmin"]


def read_size(args):
    root, rel = args
    with Image.open(resolve_image(root, rel)) as im:
        return im.size


def get_sizes(df, cache):
    if Path(cache).exists():
        c = pd.read_csv(cache)
        if c["sample_id"].tolist() == df["sample_id"].tolist():
            return c
    with ThreadPoolExecutor(8) as ex:
        sz = list(ex.map(read_size, zip(df["_image_root"], df["image_path"])))
    c = pd.DataFrame({"sample_id": df["sample_id"], "w": [s[0] for s in sz], "h": [s[1] for s in sz]})
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    c.to_csv(cache, index=False)
    return c


def size_feats(df, w, h):
    """현재/과거 행의 이미지 크기만 사용하는 피처 (같은 실험 안에서 시간순)."""
    n = len(df)
    out = {c: np.full(n, np.nan) for c in SIZE_COLS}
    t_all = df["relative_time_s"].to_numpy(float)
    for pos in exp_groups(df):
        ww, hh, t = w[pos].astype(float), h[pos].astype(float), t_all[pos]
        out["sz_w"][pos], out["sz_h"][pos] = ww, hh
        out["sz_aspect"][pos], out["sz_area"][pos] = ww / hh, ww * hh
        out["sz_dw0"][pos], out["sz_dh0"][pos] = ww - ww[0], hh - hh[0]
        chg = np.zeros(len(pos))
        chg[1:] = ((ww[1:] != ww[:-1]) | (hh[1:] != hh[:-1])).astype(float)
        out["sz_changed"][pos] = chg
        out["sz_nchg"][pos] = np.cumsum(chg)
        last = np.where(chg > 0, t, np.nan)
        last = pd.Series(last).ffill().to_numpy()
        out["sz_since_chg_s"][pos] = np.where(np.isnan(last), t - t[0], t - last)
        out["sz_wmax"][pos], out["sz_hmax"][pos] = np.maximum.accumulate(ww), np.maximum.accumulate(hh)
        out["sz_wmin"][pos], out["sz_hmin"][pos] = np.minimum.accumulate(ww), np.minimum.accumulate(hh)
    return pd.DataFrame(out)


def f1m(y, p):
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def cv(X, y, fold, rounds, name, df):
    oof = np.zeros((len(y), 6))
    for k in sorted(set(fold)):
        tr, va = fold != k, fold == k
        oof[va] = predict_gb(fit_gb(X[tr], y[tr], rounds, 100), X[va])
    pred = oof.argmax(1) + 1
    fm = [f1m(y[fold == k], pred[fold == k]) for k in sorted(set(fold))]
    print(f"\n=== {name} ===  풀링 OOF Macro F1={f1m(y, pred):.4f} | fold평균 {np.mean(fm):.4f} | 최소 {min(fm):.4f}")
    print("   fold별:", " ".join(f"{x:.3f}" for x in fm))
    print("   단계별 F1:", " ".join(f"{k}:{f1_score(y == k, pred == k):.3f}" for k in LABELS))
    for m, nm in ((1, "가열"), (2, "과충전")):
        mk = (df["tr_method"] == m).to_numpy()
        if mk.any():
            print(f"   tr_method {nm}: {f1m(y[mk], pred[mk]):.4f}")
    return f1m(y, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--folds", default="artifacts/cv/folds_exp.csv")
    ap.add_argument("--cache", default="artifacts/v2/img_sizes.csv")
    ap.add_argument("--rounds", type=int, default=300)
    a = ap.parse_args()
    D = Path(a.data_dir).expanduser()
    df = attach_folds(load_labeled(str(D / "train/train.csv"), str(D / "validation/validation.csv")), a.folds).reset_index(drop=True)
    df["fold"] = df["fold"].astype(int)
    tr_ids = set(pd.read_csv(D / "train/train.csv", usecols=["sample_id"])["sample_id"])
    df["_image_root"] = np.where(df["sample_id"].isin(tr_ids), str(D / "train"), str(D / "validation"))
    y, fold = df["label"].to_numpy(int), df["fold"].to_numpy(int)

    sz = get_sizes(df, a.cache)
    w, h = sz["w"].to_numpy(int), sz["h"].to_numpy(int)
    df["_wh"] = [f"{x}x{z}" for x, z in zip(w, h)]
    print(f"[크기] 서로 다른 (가로x세로) {df['_wh'].nunique()}종, 가장 흔한 5개:")
    print(df["_wh"].value_counts().head(5).to_string())
    print("\n[단계별 크기 분포] (행 비율 %, 상위 크기 5종)")
    top = df["_wh"].value_counts().head(5).index
    ct = pd.crosstab(df["label"], df["_wh"].where(df["_wh"].isin(top), "기타"), normalize="index") * 100
    print(ct.round(1).to_string())

    te = pd.read_csv(D / "test/test.csv")
    with ThreadPoolExecutor(8) as ex:
        tsz = list(ex.map(read_size, [(str(D / "test"), p) for p in te["image_path"]]))
    twh = pd.Series([f"{x}x{z}" for x, z in tsz])
    seen = twh.isin(set(df["_wh"]))
    print(f"\n[Test 크기] 서로 다른 {twh.nunique()}종, 학습에서 본 크기와 같은 행 비율 {seen.mean():.1%}")

    base = build_feature_table(df, None, "B")
    sf = size_feats(df, w, h)
    cv(base, y, fold, a.rounds, "센서만 (B)", df)
    cv(pd.concat([base, sf], axis=1), y, fold, a.rounds, "센서 + 이미지 크기 피처", df)
    cv(pd.concat([df[["relative_time_s"]].reset_index(drop=True), sf], axis=1), y, fold, a.rounds, "시간 + 이미지 크기만 (센서 없음)", df)
    print("\n해석: '센서만' 대비 '센서+크기'가 +0.02 이상 오르고, Test 크기가 학습과 같은 비율이 높으면 크기 정보가 실제로 쓸 수 있다는 뜻.")
    print("      크기가 실험 고유값이면(실험 안에서 거의 안 변함) '실험 구분'에 쓰이는 것이므로 Test에서의 일반화는 별개로 확인 필요.")


if __name__ == "__main__":
    main()