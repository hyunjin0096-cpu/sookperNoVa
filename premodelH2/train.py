"""학습 스크립트 v10.

예) python train.py --data_dir /home/ubuntu/data/battery_data_light --model_out model.joblib --cache_dir cache

평가 방식
  Train(78개) + Validation(19개) = 97개 실험을 합쳐 실험 단위 GroupKFold(5)로 평가합니다.
  Validation 은 5단계 비율이 Train 의 2배라 한쪽만 쓰면 점수가 흔들립니다. 합쳐야 분포가 고르게 섞입니다.
  블렌딩 가중치·피처 선택·전진 필터는 모두 이 OOF 결과로만 정하고, Test 는 쓰지 않습니다.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

import model as M


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=M.LABELS, average='macro', zero_division=0)


def load_split(data_dir: Path, split: str, cache_dir):
    frame = pd.read_csv(data_dir / split / f'{split}.csv',
                        dtype={'sample_id': 'string', 'experiment_id': 'string'})
    cache = Path(cache_dir) / f'img_{split}_{M.IMG_VERSION}.pkl' if cache_dir else None
    img = None
    if cache is not None and cache.is_file():
        img = pd.read_pickle(cache)
        img.index = img.index.astype(str)
        if set(img.index) != set(frame.sample_id.astype(str)) or list(img.columns) != M.IMAGE_STATS:
            img = None
    if img is None:
        print(f'{split}: 이미지 피처 계산 중...')
        paths = M.resolve_image_paths(frame, [data_dir / split, data_dir])
        img = M.extract_image_features(frame, paths)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            img.to_pickle(cache)
    return frame, img


def build_augmented(frame, img, y_by_id, strides):
    """원본 + 솎아낸 복제본. 복제본 피처는 솎아낸 행만으로 다시 계산합니다."""
    Xs, ys, gs, orig = [], [], [], []
    for s in strides:
        sub = frame if s == 1 else M.subsample_stride(frame, s)
        Xs.append(M.build_features(sub, img))
        ys.append(sub['sample_id'].astype(str).map(y_by_id).to_numpy(dtype=int))
        gs.append(sub['experiment_id'].to_numpy())
        orig.append(np.full(len(sub), s == 1))
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(gs), np.concatenate(orig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=Path, required=True)
    p.add_argument('--model_out', type=Path, default=Path('model.joblib'))
    p.add_argument('--cache_dir', type=Path, default=Path('cache'))
    p.add_argument('--n_estimators', type=int, default=600)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--top_n', type=int, default=120, help='사용할 피처 수 (0이면 전체)')
    p.add_argument('--folds', type=int, default=5)
    args = p.parse_args()

    tr_frame, tr_img = load_split(args.data_dir, 'train', args.cache_dir)
    va_frame, va_img = load_split(args.data_dir, 'validation', args.cache_dir)
    assert set(tr_frame.experiment_id).isdisjoint(va_frame.experiment_id)

    frame = pd.concat([tr_frame, va_frame], ignore_index=True)
    img = pd.concat([tr_img, va_img])
    y_by_id = pd.Series(frame['label'].astype(int).values, index=frame.sample_id.astype(str).values)
    print(f'합친 데이터: {len(frame):,}행 / {frame.experiment_id.nunique()}개 실험')
    print('가스 센서 컬럼:', M.gas_columns(frame))
    print(f'상대 정규화 {M.USE_RELATIVE} | 사건 피처 {M.USE_EVENTS} | CNN {M.USE_CNN} | meta {M.USE_META}')

    strides = (1,) + tuple(M.AUG_STRIDES)
    Xa, ya, ga, is_orig = build_augmented(frame, img, y_by_id, strides)
    Xo = Xa[is_orig].reset_index(drop=True)
    yo, go = ya[is_orig], ga[is_orig]
    print(f'피처 수 {Xa.shape[1]} | 학습 행 {len(Xa):,} (원본 {is_orig.sum():,})')

    # ---- 1) 피처 선택: 전체로 한 번 학습해 상위 N개만 남깁니다 ------------------------------
    features = list(Xa.columns)
    if args.top_n and args.top_n < len(features):
        probe = M._lgbm('multiclass', 'balanced', 300, M.SEED).fit(Xa, ya)
        features = M.select_features(probe, features, args.top_n)
        print(f'피처 선택: {Xa.shape[1]} → {len(features)}개')

    # ---- 2) 실험 단위 GroupKFold OOF -----------------------------------------------------
    folds = list(GroupKFold(n_splits=min(args.folds, len(np.unique(go)))).split(Xo, yo, groups=go))
    oof_multi = np.zeros((len(Xo), 6))
    oof_ord = np.zeros((len(Xo), 6))
    fold_scores = []
    for i, (tr_idx, va_idx) in enumerate(folds):
        fit_mask = np.isin(ga, np.unique(go[tr_idx]))
        multi, ordinal = M.fit_models(Xa.loc[fit_mask, features], ya[fit_mask], 'balanced',
                                      args.n_estimators, 1)
        Xv = Xo.iloc[va_idx][features]
        oof_multi[va_idx] = multi.predict_proba(Xv)
        oof_ord[va_idx] = ordinal.predict_proba(Xv)
        m = macro_f1(yo[va_idx], oof_multi[va_idx].argmax(1) + 1)
        o = macro_f1(yo[va_idx], oof_ord[va_idx].argmax(1) + 1)
        fold_scores.append((m, o))
        print(f'  fold {i + 1}/{len(folds)}: 다중분류 {m:.4f} | 순서형 {o:.4f}')

    ms = np.array([s[0] for s in fold_scores])
    os_ = np.array([s[1] for s in fold_scores])
    print(f'\nOOF 다중분류 {macro_f1(yo, oof_multi.argmax(1) + 1):.4f} (fold 평균 {ms.mean():.4f} ± {ms.std():.4f})')
    print(f'OOF 순서형   {macro_f1(yo, oof_ord.argmax(1) + 1):.4f} (fold 평균 {os_.mean():.4f} ± {os_.std():.4f})')

    # ---- 3) 블렌딩 가중치 ----------------------------------------------------------------
    w, blend_score = M.tune_blend(oof_multi, oof_ord, yo)
    print(f'블렌딩 가중치 w={w:.1f} (순서형 비중) → OOF Macro F1 {blend_score:.4f}')
    oof = M.blend(oof_multi, oof_ord, w)

    # ---- 4) 전진 필터 --------------------------------------------------------------------
    stats = M.estimate_filter_stats(frame, frame['label'].astype(int).values)
    base_frame = frame.reset_index(drop=True)
    filt = {tau: macro_f1(yo, M.causal_filter(oof, base_frame, stats, tau)) for tau in (0.5, 0.7, 1.0)}
    best_tau = max(filt, key=filt.get)
    use_filter = filt[best_tau] > blend_score + 0.003
    print('전진 필터: ' + ' | '.join(f'tau={t} {s:.4f}' for t, s in filt.items()) + f' → 사용 {use_filter}')

    final = M.blend(oof_multi, oof_ord, w)
    pred = M.causal_filter(final, base_frame, stats, best_tau) if use_filter else final.argmax(1) + 1
    print(f'\n[OOF 최종] Macro F1 {macro_f1(yo, pred):.4f}')
    print(classification_report(yo, pred, labels=M.LABELS, digits=4, zero_division=0))
    print(pd.DataFrame(confusion_matrix(yo, pred, labels=M.LABELS),
                       index=[f'실제 {l}' for l in M.LABELS], columns=[f'예측 {l}' for l in M.LABELS]))

    # 15초 간격 재현성 확인
    sub = M.subsample_stride(frame, 3)
    Xs = M.build_features(sub, img)[features]
    ys = sub['sample_id'].astype(str).map(y_by_id).to_numpy(dtype=int)
    pos = pd.Series(range(len(Xo)), index=frame.sample_id.astype(str).values)
    idx = pos.loc[sub.sample_id.astype(str).values].to_numpy()
    print(f'[참고] 15초 간격 행에 대한 OOF Macro F1 {macro_f1(ys, pred[idx]):.4f}')

    # ---- 5) 최종 학습: 전체 데이터 ---------------------------------------------------------
    print(f'\n전체 97개 실험으로 최종 학습 (시드 {args.seeds}개)...')
    multi, ordinal = M.fit_models(Xa[features], ya, 'balanced', args.n_estimators, args.seeds)
    bundle = {'multi': multi, 'ordinal': ordinal, 'features': features, 'w': w,
              'filter': {'stats': stats, 'tau': best_tau} if use_filter else None, 'cnn': None}

    gain = pd.Series(multi.booster_.feature_importance('gain'), index=features)
    gain = gain / gain.sum()
    is_img = gain.index.str.startswith('img_')
    print(f'[피처 중요도] 이미지 비중 {gain[is_img].sum():.1%} | 상위 12개:')
    print(gain.sort_values(ascending=False).head(12).round(4).to_string())

    joblib.dump(bundle, args.model_out)
    print('\n모델 저장:', args.model_out)


if __name__ == '__main__':
    main()
