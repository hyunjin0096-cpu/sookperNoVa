"""LightGBM + 확장 피처 학습 스크립트.

예) python train.py --data_dir ./data/battery_data_light --model_out ./model.joblib
    python train.py --data_dir ./data/battery_data_light --model_out ./model.joblib --refit_train_val

모델 설정(class_weight)은 Train 실험 단위 GroupKFold 점수로 고르고,
Validation은 최종 성능 확인에만 씁니다(베이스라인 노트북 권장 방식).
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold

import model as M


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=M.LABELS, average='macro', zero_division=0)


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})


def image_features(data_dir: Path, split: str, frame: pd.DataFrame, cache_dir: Path | None) -> pd.DataFrame:
    """이미지 피처는 샘플마다 독립 계산이므로 캐시해도 결과가 같습니다(round_trip으로 소수점 손실 방지)."""
    cache = cache_dir / f'img_features_v2_{split}.csv' if cache_dir else None
    if cache is not None and cache.is_file():
        img = pd.read_csv(cache, dtype={'sample_id': 'string'}, float_precision='round_trip').set_index('sample_id')
        if set(img.index) == set(frame.sample_id) and list(img.columns) == M.IMAGE_STATS:
            return img
    paths = M.resolve_image_paths(frame, [data_dir / split, data_dir])
    img = M.extract_image_features(frame, paths)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        img.to_csv(cache)
    return img


def featurize(frame, img, spec, augment_steps):
    """원본 + (선택) 간격 증강 복사본의 피처·라벨·그룹(원래 실험 ID)·원본 여부를 돌려줍니다."""
    parts = [(frame, 1)] + [(M.subsample_experiments(frame, k), k) for k in augment_steps]
    X, y, groups, step = [], [], [], []
    for f, k in parts:
        X.append(M.build_features(f, img, spec))
        y.append(pd.to_numeric(f['label'], errors='raise').astype(int))
        groups.append(f['experiment_id'].str.replace(r'__every\d+$', '', regex=True))
        step.append(pd.Series(k, index=f.index))
    cat = lambda xs: pd.concat(xs, ignore_index=True)
    return cat(X), cat(y), cat(groups), cat(step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True, help='train/, validation/ 폴더가 있는 경로')
    parser.add_argument('--model_out', type=Path, default=Path('model.joblib'))
    parser.add_argument('--refit_train_val', action='store_true', help='설정 선택 후 Train+Validation으로 다시 학습')
    parser.add_argument('--augment_steps', type=int, nargs='*', default=[2, 3],
                        help='간격 증강: 2=10초, 3=15초 간격 복사본. 끄려면 인자 없이 --augment_steps')
    parser.add_argument('--exclude', nargs='*', default=[], help='피처에서 뺄 CSV 컬럼명')
    parser.add_argument('--cache_dir', type=Path, default=Path('cache'), help='이미지 피처 캐시 폴더')
    parser.add_argument('--oof_out', type=Path, default=Path('oof_train.csv'),
                        help='선택된 설정의 Train OOF 확률 저장(다음 단계: 단조성 보정 튜닝용)')
    args = parser.parse_args()

    train = read_split(args.data_dir, 'train')
    val = read_split(args.data_dir, 'validation')
    assert set(train.experiment_id).isdisjoint(val.experiment_id)

    spec = M.infer_feature_spec(train, exclude=args.exclude)
    print('입력 컬럼 규격 (Train 기준으로 고정):')
    print(M.describe_spec(spec))

    img_tr = image_features(args.data_dir, 'train', train, args.cache_dir)
    img_va = image_features(args.data_dir, 'validation', val, args.cache_dir)
    X_tr, y_tr, g_tr, step_tr = featurize(train, img_tr, spec, args.augment_steps)
    X_va = M.build_features(val, img_va, spec)
    y_va = pd.to_numeric(val['label'], errors='raise').astype(int)
    # 15초 간격 Validation 복사본: Test의 넓은 간격에서 성능이 유지되는지 확인용
    val15 = M.subsample_experiments(val, 3)
    X_va15 = M.build_features(val15, img_va, spec)
    y_va15 = pd.to_numeric(val15['label'], errors='raise').astype(int)
    features = list(X_tr.columns)
    print(f'피처 수: {len(features)} | 학습 행: {len(X_tr):,} (원본 {int((step_tr == 1).sum()):,} + 증강)')

    n_splits = min(5, g_tr.nunique())
    orig = (step_tr == 1).to_numpy()
    results = {}
    for name, cw in [('none', None), ('balanced', 'balanced')]:
        cv, oof = [], np.full((len(X_tr), len(M.LABELS)), np.nan)
        for tr_idx, va_idx in GroupKFold(n_splits=n_splits).split(X_tr, y_tr, groups=g_tr):
            m = M.build_model(cw).fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            oof[va_idx] = m.predict_proba(X_tr.iloc[va_idx])
            ev = va_idx[orig[va_idx]]                     # 점수는 원본(5초 간격) 행으로만 계산
            cv.append(macro_f1(y_tr.iloc[ev], m.predict(X_tr.iloc[ev])))
        m = M.build_model(cw).fit(X_tr, y_tr)
        va_f1 = macro_f1(y_va, m.predict(X_va))
        va15_f1 = macro_f1(y_va15, m.predict(X_va15))
        results[name] = dict(cw=cw, cv=np.mean(cv), oof=oof, model=m)
        print(f'class_weight={name}: Train GroupKFold Macro F1 {np.mean(cv):.4f} ± {np.std(cv):.4f} '
              f'| Validation {va_f1:.4f} | Validation(15초 간격) {va15_f1:.4f}')

    best = max(results, key=lambda k: results[k]['cv'])
    cw = results[best]['cw']
    print(f'선택: class_weight={best} (Train GroupKFold 기준)')
    print('Validation 클래스별 결과:')
    print(classification_report(y_va, results[best]['model'].predict(X_va), labels=M.LABELS, digits=4, zero_division=0))

    if args.oof_out:
        oof = pd.DataFrame(results[best]['oof'][orig], columns=[f'p{l}' for l in M.LABELS])
        oof.insert(0, 'label', y_tr[orig].values)
        oof.insert(0, 'relative_time_s', X_tr.loc[orig, 'relative_time_s'].values)
        oof.insert(0, 'experiment_id', g_tr[orig].values)
        oof.insert(0, 'sample_id', train['sample_id'].values)   # featurize에서 원본이 맨 앞에 붙어 순서 동일
        oof.to_csv(args.oof_out, index=False)
        print('OOF 확률 저장:', args.oof_out)

    if args.refit_train_val:
        X_va_aug, y_va_aug, _, _ = featurize(val, img_va, spec, args.augment_steps)
        final = M.build_model(cw).fit(pd.concat([X_tr, X_va_aug], ignore_index=True),
                                      pd.concat([y_tr, y_va_aug], ignore_index=True))
    else:
        final = results[best]['model']
    joblib.dump({'model': final, 'features': features, 'spec': spec, 'class_weight': cw,
                 'params': M.LGBM_PARAMS, 'augment_steps': args.augment_steps,
                 'refit_train_val': args.refit_train_val}, args.model_out)
    print('모델 저장:', args.model_out)


if __name__ == '__main__':
    main()
