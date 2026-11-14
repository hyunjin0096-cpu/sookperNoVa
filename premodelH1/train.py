"""개선 모델 학습 스크립트.

예) python train.py --data_dir ./work/data/battery_data_light --model_out ./model.joblib --cache_dir ./work/cache

모델·설정 선택은 Train 실험 단위 GroupKFold 점수로만 합니다. Validation 은 확인용입니다.
(--refit_train_val 을 주면 선택이 끝난 뒤 Train+Validation 으로 다시 학습합니다.)
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
    frame = pd.read_csv(data_dir / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    cache = Path(cache_dir) / f'img_{split}_{M.IMG_VERSION}.pkl' if cache_dir else None
    img = None
    if cache is not None and cache.is_file():
        img = pd.read_pickle(cache)
        if set(img.index) != set(frame.sample_id):
            img = None
    if img is None:
        paths = M.resolve_image_paths(frame, [data_dir / split, data_dir])
        img = M.extract_image_features(frame, paths)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            img.to_pickle(cache)
    y = pd.to_numeric(frame['label'], errors='raise').astype(int)
    assert y.isin(M.LABELS).all()
    return frame, img, y


def make_training_set(frame, img, y_by_id, strides):
    """원본(stride 1) + 솎아낸 복제본의 피처·라벨·소속 실험을 이어 붙입니다."""
    Xs, ys, gs, orig = [], [], [], []
    for s in strides:
        sub = frame if s == 1 else M.subsample_stride(frame, s)
        Xs.append(M.build_features(sub, img))
        ys.append(sub['sample_id'].map(y_by_id).to_numpy(dtype=int))
        gs.append(sub['experiment_id'].to_numpy())
        orig.append(np.full(len(sub), s == 1))
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(gs), np.concatenate(orig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True, help='train/, validation/ 폴더가 있는 경로')
    parser.add_argument('--model_out', type=Path, default=Path('model.joblib'))
    parser.add_argument('--cache_dir', type=Path, default=None, help='이미지 피처 캐시 폴더 (다시 실행할 때 빨라집니다)')
    parser.add_argument('--refit_train_val', action='store_true')
    parser.add_argument('--n_estimators', type=int, default=300)
    args = parser.parse_args()

    train, img_tr, y_tr = load_split(args.data_dir, 'train', args.cache_dir)
    val, img_va, y_va = load_split(args.data_dir, 'validation', args.cache_dir)
    assert set(train.experiment_id).isdisjoint(val.experiment_id)
    y_tr_by_id = pd.Series(y_tr.values, index=train.sample_id.values)
    y_va_by_id = pd.Series(y_va.values, index=val.sample_id.values)

    print('가스 센서로 잡힌 컬럼:', M.gas_columns(train))
    print('thermal_t 사용:', M.USE_THERMAL_T, '| 실험조건(meta) 사용:', M.USE_META, '| 누적 최대/최소 사용:', M.USE_CUMULATIVE)

    strides = (1,) + tuple(M.AUG_STRIDES)
    Xa, ya, ga, is_orig = make_training_set(train, img_tr, y_tr_by_id, strides)
    print(f'피처 수 {Xa.shape[1]} | 학습 행 {len(Xa):,} (원본 {is_orig.sum():,} + 솎아낸 복제본)')
    Xo, yo, go = Xa[is_orig].reset_index(drop=True), ya[is_orig], ga[is_orig]

    # ---- 1) 클래스 가중치 선택: Train 실험 단위 교차검증 -------------------------------------
    n_splits = min(5, len(np.unique(go)))
    folds = list(GroupKFold(n_splits=n_splits).split(Xo, yo, groups=go))
    cv_results, oof_store = {}, {}
    for name, cw in [('none', None), ('balanced', 'balanced')]:
        oof = np.zeros((len(Xo), 6))
        scores = []
        for tr_idx, va_idx in folds:
            tr_exps = set(go[tr_idx])
            fit_mask = np.isin(ga, list(tr_exps))
            m = M.build_model(cw, args.n_estimators).fit(Xa[fit_mask], ya[fit_mask])
            oof[va_idx] = m.predict_proba(Xo.iloc[va_idx])
            scores.append(macro_f1(yo[va_idx], m.classes_[oof[va_idx].argmax(1)]))
        cv_results[name] = (cw, float(np.mean(scores)), float(np.std(scores)))
        oof_store[name] = oof
        print(f'class_weight={name}: Train GroupKFold Macro F1 {np.mean(scores):.4f} ± {np.std(scores):.4f}')
    best = max(cv_results, key=lambda k: cv_results[k][1])
    cw = cv_results[best][0]
    print('선택(CV 기준):', best)

    # ---- 2) 전진 필터 사용 여부: OOF 확률로 tau 선택, 이득이 작으면 사용하지 않음 -----------------
    classes = np.arange(1, 7)
    plain = macro_f1(yo, classes[oof_store[best].argmax(1)])
    stats_tr = M.estimate_filter_stats(train, y_tr)
    filt_scores = {}
    for tau in (0.3, 0.5, 0.7, 1.0):
        pred = M.causal_filter(oof_store[best], train.reset_index(drop=True), stats_tr, tau)
        # Xo 의 행 순서는 train 의 행 순서와 같습니다 (stride 1 이 먼저 이어 붙여짐).
        filt_scores[tau] = macro_f1(yo, pred)
    best_tau = max(filt_scores, key=filt_scores.get)
    print(f'OOF Macro F1: 필터 없음 {plain:.4f} | ' + ' | '.join(f'tau={t}: {s:.4f}' for t, s in filt_scores.items()))
    use_filter = filt_scores[best_tau] > plain + 0.003
    print('전진 필터 사용:', use_filter, f'(tau={best_tau})' if use_filter else '')

    # ---- 3) Validation 확인 (선택에는 쓰지 않음) -----------------------------------------------
    model = M.build_model(cw, args.n_estimators).fit(Xa, ya)
    bundle = {'pipeline': model, 'features': list(Xa.columns), 'class_weight': cw,
              'filter': {'stats': stats_tr, 'tau': best_tau} if use_filter else None}
    for label, vframe in [('stride 1 (5초)', val), ('stride 3 (15초)', M.subsample_stride(val, 3))]:
        Xv = M.build_features(vframe, img_va)
        yv = vframe['sample_id'].map(y_va_by_id).to_numpy(dtype=int)
        pred = M.predict_labels(bundle, vframe, Xv)
        print(f'\n[Validation {label}] Macro F1 {macro_f1(yv, pred):.4f}')
        if label.startswith('stride 1'):
            print(classification_report(yv, pred, labels=M.LABELS, digits=4, zero_division=0))
            print(pd.DataFrame(confusion_matrix(yv, pred, labels=M.LABELS),
                               index=[f'실제 {l}' for l in M.LABELS], columns=[f'예측 {l}' for l in M.LABELS]))

    # ---- 4) 최종 모델 ------------------------------------------------------------------------
    if args.refit_train_val:
        Xv_all, yv_all, _, _ = make_training_set(val, img_va, y_va_by_id, strides)
        Xf, yf = pd.concat([Xa, Xv_all], ignore_index=True), np.concatenate([ya, yv_all])
        model = M.build_model(cw, args.n_estimators).fit(Xf, yf)
        both = pd.concat([train, val], ignore_index=True)
        stats_all = M.estimate_filter_stats(both, np.concatenate([y_tr.values, y_va.values]))
        bundle = {'pipeline': model, 'features': list(Xf.columns), 'class_weight': cw,
                  'filter': {'stats': stats_all, 'tau': best_tau} if use_filter else None}
        print('\nTrain+Validation 으로 다시 학습했습니다.')
    # 이미지 피처가 실제로 쓰이는지 확인 (규정: 이미지와 센서를 모두 활용해야 함)
    gain = pd.Series(bundle['pipeline'].booster_.feature_importance('gain'), index=bundle['features'])
    gain = gain / gain.sum()
    is_img = gain.index.str.startswith('img_')
    print(f'\n[피처 중요도] 이미지 관련 피처 비중 {gain[is_img].sum():.1%} | 상위 10개:')
    print(gain.sort_values(ascending=False).head(10).round(4).to_string())
    joblib.dump(bundle, args.model_out)
    print('모델 저장:', args.model_out)


if __name__ == '__main__':
    main()
