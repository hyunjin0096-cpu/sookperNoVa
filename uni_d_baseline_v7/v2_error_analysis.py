"""v2_error_analysis.py - OOF 예측이 '어디서' 틀리는지 분석 (재학습 없음, 몇 초)

  1) 혼동행렬(어느 단계끼리 헷갈리는가)
  2) 조건별 점수(가열/과충전, 배터리 종류, SOC, 용량)
  3) 가장 못 맞히는 실험들
  4) 오류가 '단계 경계 근처'(시점 어긋남)인지 '경계와 먼 곳'(실험 전체 오판)인지
  5) 예측 단계가 시간에 따라 줄어드는(되돌아가는) 비율 - 라벨은 줄어들지 않음

사용:  python -u -m v2_error_analysis --data-dir "$D" --oof artifacts/v2/final_oof_blend.csv 2>&1 | tee v2_err.log
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

L = [1, 2, 3, 4, 5, 6]
P = [f"prob_{k}" for k in L]


def f1m(y, p):
    return f1_score(y, p, labels=L, average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--oof", default="artifacts/v2/final_oof_blend.csv")
    a = ap.parse_args()
    D = Path(a.data_dir).expanduser()
    cols = ["sample_id", "experiment_id", "relative_time_s", "label", "tr_method", "bat_class", "soc_status", "bat_cc"]
    t = pd.concat([pd.read_csv(D / "train/train.csv", usecols=cols), pd.read_csv(D / "validation/validation.csv", usecols=cols)])
    o = pd.read_csv(a.oof)
    df = t.merge(o[["sample_id"] + P], on="sample_id", validate="one_to_one")
    df["pred"] = df[P].to_numpy().argmax(1) + 1
    df = df.sort_values(["experiment_id", "relative_time_s", "sample_id"], kind="stable").reset_index(drop=True)
    y, p = df["label"].to_numpy(), df["pred"].to_numpy()
    print(f"행 {len(df):,}, 실험 {df.experiment_id.nunique()}, 풀링 Macro F1={f1m(y, p):.4f}, 정확도={(y == p).mean():.4f}\n")

    print("[1] 혼동행렬 (행=실제, 열=예측, 행 기준 %)")
    cm = pd.crosstab(df["label"], df["pred"], normalize="index") * 100
    print(cm.reindex(index=L, columns=L, fill_value=0).round(1).to_string())

    print("\n[2] 조건별 (행 수, 정확도, Macro F1)")
    for c in ["tr_method", "bat_class", "soc_status", "bat_cc"]:
        rows = []
        for v, g in df.groupby(c):
            rows.append((c, v, g.experiment_id.nunique(), len(g), (g.label == g.pred).mean(), f1m(g.label, g.pred)))
        print(pd.DataFrame(rows, columns=["조건", "값", "실험수", "행수", "정확도", "MacroF1"]).round(3).to_string(index=False))

    print("\n[3] 정확도가 낮은 실험 15개")
    e = df.groupby("experiment_id").apply(lambda g: pd.Series(dict(
        행수=len(g), 정확도=(g.label == g.pred).mean(), tr_method=g.tr_method.iloc[0], bat_class=g.bat_class.iloc[0],
        soc=g.soc_status.iloc[0], cc=g.bat_cc.iloc[0], 평균단계오차=(g.pred - g.label).abs().mean())), include_groups=False)
    print(e.sort_values("정확도").head(15).round(3).to_string())
    print(f"   실험별 정확도: 중앙값 {e['정확도'].median():.3f}, 0.8 미만 실험 {int((e['정확도'] < 0.8).sum())}개 / {len(e)}개")

    print("\n[4] 오류의 위치 (경계 = 실제 단계가 바뀌는 시점)")
    dist = np.full(len(df), np.nan)
    for _, idx in df.groupby("experiment_id").indices.items():
        lab, tt = y[idx], df["relative_time_s"].to_numpy()[idx]
        ch = tt[1:][lab[1:] != lab[:-1]]
        dist[idx] = np.abs(tt[:, None] - ch[None, :]).min(1) if len(ch) else np.nan
    err = p != y
    d = dist[err]
    d = d[~np.isnan(d)]
    for lo, hi in [(0, 15), (15, 30), (30, 60), (60, 120), (120, 1e9)]:
        print(f"   경계에서 {lo:>3}~{hi if hi < 1e9 else '∞'}초: 오류 {((d >= lo) & (d < hi)).sum():>5} ({((d >= lo) & (d < hi)).mean():.1%})")
    adj = (np.abs(p - y) == 1)[err].mean()
    print(f"   오류 중 인접 단계(±1) 비율 {adj:.1%}, 2단계 이상 차이 {1 - adj:.1%}")

    print("\n[5] 예측 단계의 시간 역행 (라벨은 줄어들지 않음)")
    back, tot = 0, 0
    for _, idx in df.groupby("experiment_id").indices.items():
        dp = np.diff(p[idx])
        back += int((dp < 0).sum())
        tot += len(dp)
    print(f"   연속한 두 행에서 예측 단계가 내려가는 비율 {back / tot:.2%} ({back:,}/{tot:,})")
    print("\n해석: 오류가 경계 60초 안에 몰려 있으면 '시점 정밀도' 문제(이벤트 타이밍 피처로 개선), 경계와 멀면 '실험 전체 오판'(조건/도메인 문제).")


if __name__ == "__main__":
    main()