"""v5 학습 스크립트: v4(U-Net 모양 피처) + 물리 기반 지표(기본 사용, --no_physics로 끔).

예) python train.py --data_dir ./data/battery_data_light --model_out ./model.joblib
    python train.py ... --algo catboost --class_weights balanced     # CatBoost 비교
    python train.py ... --no_interactions                             # 상호작용 피처 효과 검증(끄고 비교)
    python train.py ... --no_interactions --unet unet.pt              # v4: U-Net 모양 피처 추가

- 모델 설정(class_weight)은 Train 실험 단위 GroupKFold 점수로 고릅니다.
- 단조성 보정 파라미터(alpha, down_eps)는 Train OOF 확률로 고르고, 전이행렬은 Train 라벨로 추정합니다.
- Validation은 최종 확인에만 씁니다.
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


def per_class_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=M.LABELS, average=None, zero_division=0)


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


def unet_features(data_dir: Path, split: str, frame: pd.DataFrame, weights: Path, cache_dir: Path | None,
                  device: str) -> pd.DataFrame:
    """U-Net 모양 피처(캐시 파일명에 가중치 파일 이름·수정 시각을 넣어 가중치가 바뀌면 다시 계산)."""
    import unet as U
    tag = f'{weights.stem}_{int(weights.stat().st_mtime)}'
    cache = cache_dir / f'unet_features_{tag}_{split}.csv' if cache_dir else None
    if cache is not None and cache.is_file():
        f = pd.read_csv(cache, dtype={'sample_id': 'string'}, float_precision='round_trip').set_index('sample_id')
        if set(f.index) == set(frame.sample_id) and list(f.columns) == M.UNET_STATS:
            return f
    paths = M.resolve_image_paths(frame, [data_dir / split, data_dir])
    f = U.extract_unet_features(frame, paths, weights, device=device)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        f.to_csv(cache)
    return f


def featurize(frame, img, spec, augment_steps):
    """원본 + (선택) 간격 증강 복사본의 피처·라벨·그룹(원래 실험 ID)·간격 배수·피처용 실험 ID·시간."""
    parts = [(frame, 1)] + [(M.subsample_experiments(frame, k), k) for k in augment_steps]
    X, y, groups, step = [], [], [], []
    for f, k in parts:
        X.append(M.build_features(f, img, spec))
        y.append(pd.to_numeric(f['label'], errors='raise').astype(int))
        groups.append(f['experiment_id'].str.replace(r'__every\d+$', '', regex=True))
        step.append(pd.Series(k, index=f.index))
    cat = lambda xs: pd.concat(xs, ignore_index=True)
    return cat(X), cat(y), cat(groups), cat(step)


def show(name, y, raw_pred, pp_pred=None):
    line = f'{name}: Macro F1 {macro_f1(y, raw_pred):.4f}'
    if pp_pred is not None:
        line += f' → 단조성 보정 후 {macro_f1(y, pp_pred):.4f}'
    print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True, help='train/, validation/ 폴더가 있는 경로')
    parser.add_argument('--model_out', type=Path, default=Path('model.joblib'))
    parser.add_argument('--algo', choices=['lgbm', 'catboost'], default='lgbm')
    parser.add_argument('--class_weights', nargs='+', choices=['none', 'balanced'], default=['none', 'balanced'],
                        help='비교할 class_weight. CatBoost는 느리므로 하나만 주는 것을 권장')
    parser.add_argument('--cb_iterations', type=int, default=None, help='CatBoost 반복 수(기본 800). 빠른 비교 시 300 등')
    parser.add_argument('--cb_gpu', action='store_true',
                        help='CatBoost를 GPU로 학습(빠르지만 실행마다 미세하게 다를 수 있어 최종 제출 전 재현성 확인 필요)')
    parser.add_argument('--unet', type=Path, default=None, help='train_unet.py로 학습한 U-Net 가중치(주면 모양 피처 추가)')
    parser.add_argument('--unet_device', default='cpu', help='U-Net 피처 추출 장치. 재현성을 위해 cpu 권장')
    parser.add_argument('--no_physics', action='store_true', help='물리 기반 지표 끄기')
    parser.add_argument('--no_interactions', action='store_true', help='센서 간 상호작용 피처 끄기(효과 비교용)')
    parser.add_argument('--no_postprocess', action='store_true', help='단조성 보정 끄기')
    parser.add_argument('--refit_train_val', action='store_true', help='설정 선택 후 Train+Validation으로 다시 학습')
    parser.add_argument('--augment_steps', type=int, nargs='*', default=[2, 3],
                        help='간격 증강: 2=10초, 3=15초 간격 복사본. 끄려면 인자 없이 --augment_steps')
    parser.add_argument('--exclude', nargs='*', default=[], help='피처에서 뺄 CSV 컬럼명')
    parser.add_argument('--cache_dir', type=Path, default=Path('cache'), help='이미지 피처 캐시 폴더')
    parser.add_argument('--oof_out', type=Path, default=Path('oof_train.csv'), help='선택된 설정의 Train OOF 확률 저장')
    args = parser.parse_args()

    overrides = {}
    if args.algo == 'catboost':
        if args.cb_iterations:
            overrides['iterations'] = args.cb_iterations
        if args.cb_gpu:
            overrides.update(task_type='GPU', devices='0')
    build = lambda cw: M.build_model(cw, args.algo, **overrides)

    train = read_split(args.data_dir, 'train')
    val = read_split(args.data_dir, 'validation')
    assert set(train.experiment_id).isdisjoint(val.experiment_id)

    spec = M.infer_feature_spec(train, exclude=args.exclude, interactions=not args.no_interactions,
                                unet=args.unet is not None, physics=not args.no_physics)
    print(f'모델: {args.algo} | 입력 컬럼 규격 (Train 기준으로 고정):')
    print(M.describe_spec(spec))

    img_tr = image_features(args.data_dir, 'train', train, args.cache_dir)
    img_va = image_features(args.data_dir, 'validation', val, args.cache_dir)
    if args.unet is not None:
        img_tr = img_tr.join(unet_features(args.data_dir, 'train', train, args.unet, args.cache_dir, args.unet_device))
        img_va = img_va.join(unet_features(args.data_dir, 'validation', val, args.unet, args.cache_dir, args.unet_device))
    X_tr, y_tr, g_tr, step_tr = featurize(train, img_tr, spec, args.augment_steps)
    X_va = M.build_features(val, img_va, spec)
    y_va = pd.to_numeric(val['label'], errors='raise').astype(int)
    val15 = M.subsample_experiments(val, 3)                  # Test의 넓은 간격에서도 유지되는지 확인용
    X_va15 = M.build_features(val15, img_va, spec)
    y_va15 = pd.to_numeric(val15['label'], errors='raise').astype(int)
    features = list(X_tr.columns)
    n_ix = sum(c.startswith('ix_') for c in features)
    n_un = sum(c.startswith('unet_') for c in features)
    n_ph = sum(c.startswith('ph_') for c in features)
    print(f'피처 수: {len(features)} (상호작용 ix_ {n_ix}개, U-Net unet_ {n_un}개, 물리 ph_ {n_ph}개) | 학습 행: {len(X_tr):,} (원본 {int((step_tr == 1).sum()):,} + 증강)')

    n_splits = min(5, g_tr.nunique())
    orig = (step_tr == 1).to_numpy()
    results = {}
    for name in args.class_weights:
        cw = None if name == 'none' else 'balanced'
        cv, oof = [], np.full((len(X_tr), len(M.LABELS)), np.nan)
        for tr_idx, va_idx in GroupKFold(n_splits=n_splits).split(X_tr, y_tr, groups=g_tr):
            m = build(cw).fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            oof[va_idx] = m.predict_proba(X_tr.iloc[va_idx])
            ev = va_idx[orig[va_idx]]                     # 점수는 원본(5초 간격) 행으로만 계산
            cv.append(macro_f1(y_tr.iloc[ev], m.predict(X_tr.iloc[ev])))
        m = build(cw).fit(X_tr, y_tr)
        results[name] = dict(cw=cw, cv=np.mean(cv), oof=oof, model=m)
        print(f'class_weight={name}: Train GroupKFold Macro F1 {np.mean(cv):.4f} ± {np.std(cv):.4f} '
              f'| Validation {macro_f1(y_va, m.predict(X_va)):.4f} | Validation(15초 간격) {macro_f1(y_va15, m.predict(X_va15)):.4f}')

    best = max(results, key=lambda k: results[k]['cv'])
    cw, model = results[best]['cw'], results[best]['model']
    print(f'선택: class_weight={best} (Train GroupKFold 기준)')

    # ---- 단조성 보정: 전이행렬은 Train 라벨, alpha·down_eps는 Train OOF로 선택 ----
    pp = None
    oof = results[best]['oof'][orig]
    oof_t = X_tr.loc[orig, 'relative_time_s'].to_numpy()
    oof_g = g_tr[orig].to_numpy()
    oof_y = y_tr[orig].to_numpy()
    if not args.no_postprocess:
        trans_info = M.estimate_transitions(train)
        pp, pp_f1, table = M.tune_postprocess(oof, oof_y, oof_g, oof_t, trans_info)
        raw_f1 = macro_f1(oof_y, np.array(M.LABELS)[oof.argmax(axis=1)])
        print(f'\n[단조성 보정] Train OOF Macro F1 {raw_f1:.4f} → {pp_f1:.4f} (alpha={pp["alpha"]}, down_eps={pp["down_eps"]})')
        print('  5초 전이행렬(행=현재, 열=다음, 1~6단계):')
        print(pd.DataFrame(pp['trans'], index=M.LABELS, columns=M.LABELS).round(4).to_string())
        if pp_f1 <= raw_f1:
            print('  → Train OOF에서 개선이 없어 단조성 보정을 끕니다.')
            pp = None

    def predict_both(X, frame):
        proba = model.predict_proba(X)
        raw = np.array(M.LABELS)[proba.argmax(axis=1)]
        if pp is None:
            return raw, None
        post = np.array(M.LABELS)[M.forward_filter(proba, frame['experiment_id'].to_numpy(),
                                                   X['relative_time_s'].to_numpy(), pp).argmax(axis=1)]
        return raw, post

    print('\n[Validation 결과]')
    raw_va, pp_va = predict_both(X_va, val)
    raw_15, pp_15 = predict_both(X_va15, val15)
    show('Validation', y_va, raw_va, pp_va)
    show('Validation(15초 간격)', y_va15, raw_15, pp_15)
    final_va = pp_va if pp_va is not None else raw_va
    table = pd.DataFrame({'보정 전 F1': per_class_f1(y_va, raw_va)}, index=M.LABELS)
    if pp_va is not None:
        table['보정 후 F1'] = per_class_f1(y_va, pp_va)
    print(table.round(4).to_string())
    print(classification_report(y_va, final_va, labels=M.LABELS, digits=4, zero_division=0))
    print(pd.DataFrame(confusion_matrix(y_va, final_va, labels=M.LABELS),
                       index=[f'실제{l}' for l in M.LABELS], columns=[f'예측{l}' for l in M.LABELS]))

    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print('\n중요도 상위 20개 피처:')
    print(imp.head(20).round(1).to_string())
    if n_ix:
        rank = imp.rank(ascending=False)
        print(f'상호작용 피처 중 최고 순위: {int(rank[[c for c in features if c.startswith("ix_")]].min())}위 / {len(features)}')
    rank = imp.rank(ascending=False)
    if n_ph:
        ph = [c for c in features if c.startswith('ph_')]
        print(f'물리 지표 중 최고 순위: {int(rank[ph].min())}위 | 물리 지표 중요도 합 비율 {imp[ph].sum() / imp.sum():.3f}')
        print('물리 지표 상위 10개:'); print(imp[ph].sort_values(ascending=False).head(10).round(1).to_string())
    if n_un:
        un = [c for c in features if c.startswith('unet_')]
        print(f'U-Net 피처 중 최고 순위: {int(rank[un].min())}위 / {len(features)} | U-Net 피처 중요도 합 비율 {imp[un].sum() / imp.sum():.3f}')

    if args.oof_out and str(args.oof_out) not in ('', '.'):
        out = pd.DataFrame(oof, columns=[f'p{l}' for l in M.LABELS])
        out.insert(0, 'label', oof_y)
        out.insert(0, 'relative_time_s', oof_t)
        out.insert(0, 'experiment_id', oof_g)
        out.insert(0, 'sample_id', train['sample_id'].values)   # 원본이 featurize 결과의 맨 앞에 같은 순서로 있음
        out.to_csv(args.oof_out, index=False)
        print('OOF 확률 저장:', args.oof_out)

    if args.refit_train_val:
        X_va_aug, y_va_aug, _, _ = featurize(val, img_va, spec, args.augment_steps)
        model = build(cw).fit(pd.concat([X_tr, X_va_aug], ignore_index=True),
                                                 pd.concat([y_tr, y_va_aug], ignore_index=True))
    joblib.dump({'model': model, 'features': features, 'spec': spec, 'class_weight': cw, 'algo': args.algo,
                 'postprocess': pp, 'unet_weights': args.unet.name if args.unet else None, 'augment_steps': args.augment_steps,
                 'refit_train_val': args.refit_train_val}, args.model_out)
    print('모델 저장:', args.model_out)


if __name__ == '__main__':
    main()
