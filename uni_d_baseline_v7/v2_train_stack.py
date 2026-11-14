"""v2_train_stack.py - 2단계: 센서 인과 피처 + 이미지 모델 확률 요약 -> LightGBM. CV 비교, 최종 모델 저장.

variant (같은 fold, 같은 평가 방식으로 비교)
  A 센서 기본(이미지 없음) | B 센서 기본+보조(이미지 없음) | C 센서 기본+보조+이미지 | D 센서 기본+이미지
  A/B는 '이미지가 얼마나 점수를 올리는지' 재는 기준선이고 제출용이 아님 (제출은 이미지 사용 variant: C 또는 D)

스태킹 주의: 이미지 확률은 fold 밖 예측(OOF)이라 라벨 누수는 없지만, 2단계 CV는 근사 평가라 약간 낙관적일 수 있음.

사용(레포 루트):
  python -u -m v2_train_stack --data-dir "$D" --folds artifacts/cv/folds_exp.csv 2>&1 | tee v2_stack.log
  (이미지 OOF가 아직 없으면 --variants A B 만 먼저 실행해 기준선 확인 가능)
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from cv.common import attach_folds, load_labeled
from v2_common import (LABELS, VARIANTS, VERSION, build_feature_table, causal_filter, fit_gb, predict_gb,
                       predict_models, save_models, write_json, HAVE_LGB)

PCOLS = [f"prob_{k}" for k in LABELS]


def f1m(y, p):
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def causality_check(df, P, variant, n_exp=6, seed=0):
    """실험마다 뒤 40%를 지워도 앞 60% 행의 피처가 같아야 한다 + 행 순서를 섞어도 같아야 한다."""
    rng = np.random.default_rng(seed)
    exps = rng.choice(df["experiment_id"].unique(), size=min(n_exp, df["experiment_id"].nunique()), replace=False)
    sub = df[df["experiment_id"].isin(exps)].reset_index(drop=True)
    Ps = None if P is None else P[df["experiment_id"].isin(exps).to_numpy()]
    full = build_feature_table(sub, Ps, variant)
    s = sub.sort_values(["experiment_id", "relative_time_s", "sample_id"], kind="stable")
    keep = s[s.groupby("experiment_id").cumcount() < s.groupby("experiment_id")["sample_id"].transform("size") * 0.6].index
    part = build_feature_table(sub.loc[keep].reset_index(drop=True), None if Ps is None else Ps[keep], variant)
    a = full.loc[keep].reset_index(drop=True)
    d1 = np.nanmax(np.abs(a.to_numpy() - part.to_numpy()))
    perm = rng.permutation(len(sub))
    sh = build_feature_table(sub.iloc[perm].reset_index(drop=True), None if Ps is None else Ps[perm], variant)
    back = sh.iloc[np.argsort(perm)].reset_index(drop=True)
    d2 = np.nanmax(np.abs(full.to_numpy() - back.to_numpy()))
    ok = d1 < 1e-9 and d2 < 1e-9
    print(f"[인과성 점검 {variant}] 뒤 40% 제거 최대차이={d1:.2e}, 행 섞기 최대차이={d2:.2e} -> {'OK' if ok else '실패'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--folds", default="artifacts/cv/folds_exp.csv")
    ap.add_argument("--img-oof", default="artifacts/v2/img_oof.csv")
    ap.add_argument("--out-dir", default="artifacts/v2")
    ap.add_argument("--weights-dir", default="weights_v2")
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--final-variant", default="B", choices=["B", "C", "D"])
    ap.add_argument("--blend-w", type=float, default=0.3,
                    help="최종 확률 = (1-w) * LightGBM + w * 이미지모델 확률. B처럼 이미지 피처가 없는 구성은 w>0이어야 제출 가능")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--cv-seeds", type=int, default=1)
    ap.add_argument("--final-seeds", type=int, default=3)
    ap.add_argument("--use-filter", action="store_true", help="인과 필터를 최종 모델에 포함(운영진 허용 확인 후에만)")
    ap.add_argument("--no-final", action="store_true", help="CV 비교만 하고 최종 모델은 저장하지 않음")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    D = Path(a.data_dir).expanduser()
    df = attach_folds(load_labeled(str(D / "train/train.csv"), str(D / "validation/validation.csv")), a.folds)
    df = df.reset_index(drop=True)
    df["fold"] = df["fold"].astype(int)
    y = df["label"].to_numpy(int)
    fold = df["fold"].to_numpy(int)
    out_dir, wdir = Path(a.out_dir), Path(a.weights_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rounds = 20 if a.smoke else a.rounds
    print(f"rows={len(df):,}, experiments={df['experiment_id'].nunique()}, lightgbm={'예' if HAVE_LGB else '아니오(sklearn 대체)'}", flush=True)

    P = None
    if any(VARIANTS[v][1] for v in a.variants) or not a.no_final:
        o = pd.read_csv(a.img_oof).set_index("sample_id")
        miss = set(df["sample_id"]) - set(o.index)
        if miss:
            raise SystemExit(f"이미지 OOF에 없는 sample_id {len(miss)}개: img_oof.csv가 전체 fold 결과인지 확인하세요")
        P = o.loc[df["sample_id"], PCOLS].to_numpy(float)
        print(f"[이미지 단독 OOF] Macro F1={f1m(y, P.argmax(1) + 1):.4f}", flush=True)

    feats = {}
    for v in a.variants:
        feats[v] = build_feature_table(df, P if VARIANTS[v][1] else None, v)
        print(f"[피처] variant {v}: {feats[v].shape[1]}개", flush=True)
    for v in a.variants:
        if not causality_check(df, P if VARIANTS[v][1] else None, v):
            raise SystemExit(f"variant {v} 피처가 미래 정보를 쓰거나 행 순서에 의존합니다. 중단합니다.")

    oofs, summary = {}, []
    for v in a.variants:
        X = feats[v]
        oof = np.zeros((len(df), 6))
        for k in sorted(df["fold"].unique()):
            tr, va = fold != k, fold == k
            ps = [predict_gb(fit_gb(X[tr], y[tr], rounds, 100 + s), X[va]) for s in range(a.cv_seeds)]
            oof[va] = np.mean(ps, axis=0)
        oofs[v] = oof
        pred = oof.argmax(1) + 1
        fm = [f1m(y[fold == k], pred[fold == k]) for k in sorted(df["fold"].unique())]
        pooled = f1m(y, pred)
        summary.append((v, pooled, np.mean(fm), np.std(fm, ddof=1) if len(fm) > 1 else 0.0, min(fm)))
        print(f"\n=== variant {v} ===  풀링 OOF Macro F1={pooled:.4f} | fold 평균 {np.mean(fm):.4f} ± {summary[-1][3]:.4f} | 최소 {min(fm):.4f}", flush=True)
        print("   fold별:", " ".join(f"{x:.3f}" for x in fm))
        print("   단계별 F1:", " ".join(f"{k}:{f1_score(y == k, pred == k):.3f}" for k in LABELS))
        for m, nm in ((1, "가열"), (2, "과충전")):
            mk = (df["tr_method"] == m).to_numpy()
            if mk.any():
                print(f"   tr_method {nm}: {f1m(y[mk], pred[mk]):.4f}")
        r = pd.DataFrame(oof, columns=PCOLS)
        r.insert(0, "sample_id", df["sample_id"])
        r.insert(1, "fold", fold)
        r.to_csv(out_dir / f"stack_oof_{v}.csv", index=False)

    print("\n=== 요약 (풀링 OOF Macro F1) ===")
    for v, p, m, s, mn in summary:
        print(f"  variant {v}: {p:.4f}   (fold 평균 {m:.4f}, 표준편차 {s:.4f}, 최소 {mn:.4f})")
    d = {v: p for v, p, *_ in summary}
    if "A" in d and "D" in d:
        print(f"  이미지 효과 D-A = {d['D'] - d['A']:+.4f}")
    if "B" in d and "C" in d:
        print(f"  이미지 효과 C-B = {d['C'] - d['B']:+.4f}")
    if "A" in d and "B" in d:
        print(f"  보조 센서 피처 효과 B-A = {d['B'] - d['A']:+.4f}")
    print("  (이미지 효과가 +0.005 미만이면 이미지는 점수에 거의 기여하지 않는 것. 그래도 제출은 이미지 사용 variant여야 함)")

    final_v = a.final_variant
    if not a.no_final and not VARIANTS[final_v][1] and a.blend_w <= 0:
        raise SystemExit("이 구성은 이미지를 전혀 쓰지 않습니다. --blend-w를 0보다 크게 주거나 C/D를 선택하세요")
    if final_v not in oofs and not a.no_final:
        raise SystemExit(f"--final-variant {final_v}를 --variants에 포함시켜야 합니다")
    filt = None
    if final_v in oofs and P is not None:
        bl = (1 - a.blend_w) * oofs[final_v] + a.blend_w * P
        pr = bl.argmax(1) + 1
        fmb = [f1m(y[fold == k], pr[fold == k]) for k in sorted(df["fold"].unique())]
        print(f"\n[최종 구성 OOF] variant {final_v} + 이미지 확률 비중 {a.blend_w}: 풀링 Macro F1={f1m(y, pr):.4f} | "
              f"fold 평균 {np.mean(fmb):.4f} | 최소 {min(fmb):.4f}", flush=True)
        pd.DataFrame(bl, columns=PCOLS).assign(sample_id=df["sample_id"], fold=fold).to_csv(out_dir / "final_oof_blend.csv", index=False)
    if final_v in oofs:
        prior = np.array([(y == k).mean() for k in LABELS])
        base = f1m(y, oofs[final_v].argmax(1) + 1)
        best = None
        print(f"\n=== 인과 필터 평가 (variant {final_v}, 필터 없음 F1={base:.4f}) ===", flush=True)
        for lam, gamma, skip in itertools.product([0.0005, 0.002, 0.005], [0.5, 1.0], [0.1, 0.3]):
            pf = causal_filter(df, oofs[final_v], lam, gamma, skip, prior)
            s = f1m(y, pf.argmax(1) + 1)
            if best is None or s > best[0]:
                best = (s, lam, gamma, skip)
        print(f"  최고 {best[0]:.4f} (lambda={best[1]}, gamma={best[2]}, skip={best[3]}) -> 변화 {best[0] - base:+.4f}")
        if a.use_filter:
            filt = dict(lam=best[1], gamma=best[2], skip=best[3], prior=prior.tolist())
            print("  --use-filter: 최종 모델에 필터를 포함합니다 (운영진 허용 확인 필요)")
        else:
            print("  필터는 기본 꺼짐. 운영진이 허용하고 이득이 +0.01 이상일 때만 --use-filter로 다시 실행")

    if not a.no_final:
        X = feats[final_v]
        seeds = [10 + s for s in range(1 if a.smoke else a.final_seeds)]
        models = [fit_gb(X, y, rounds, s) for s in seeds]
        backend = save_models(models, wdir)
        write_json(wdir / "stack_config.json", dict(version=VERSION, variant=final_v, backend=backend, n_models=len(models),
                                                    feature_columns=list(X.columns), rounds=rounds, seeds=seeds, filter=filt,
                                                    blend_w=float(a.blend_w)))
        print(f"\n[저장] {wdir}/stack_config.json, stack_*.{'txt' if backend == 'lgb' else 'joblib'} (variant {final_v}, 피처 {X.shape[1]}개)")
        print("       이미지 가중치(img_fold*.pt, img_config.json)와 같은 폴더에 있어야 합니다 -> 최종 제출 때 weights/ 아래로 복사")
    print("FINISHED")


if __name__ == "__main__":
    main()