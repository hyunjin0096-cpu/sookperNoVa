"""v4: retain EXACT v3-noix 190 columns, compare frozen DINOv2 embedding variants.

Usage: python train_v4.py --data_dir ./data/battery_data_light --baseline_model ./model_v3_noix.joblib
PCA is fitted only on each fold's ORIGINAL training experiments, not validation fold.
All model selection is based on Train GroupKFold; external Validation is for final confirmation.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold
import model as M
from dino_features import extract_embeddings, DIM

LABELS = M.LABELS


def read_split(root, split):
    return pd.read_csv(root / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})


def score(y, pred):
    return float(f1_score(y, pred, labels=LABELS, average='macro', zero_division=0))


def class_scores(y, pred):
    return dict(zip(map(str, LABELS), map(float, f1_score(y, pred, labels=LABELS,
                                                       average=None, zero_division=0))))


def features_by_split(frame, img, spec, emb, augment_steps):
    parts = [(frame, 1)] + [(M.subsample_experiments(frame, k), k) for k in augment_steps]
    x_parts, e_parts, y_parts, group_parts, steps = [], [], [], [], []
    for sub, k in parts:
        x_parts.append(M.build_features(sub, img, spec).reset_index(drop=True))
        e_parts.append(emb.loc[sub['sample_id'].astype(str).to_numpy()].reset_index(drop=True))
        y_parts.append(pd.to_numeric(sub['label']).astype(int).reset_index(drop=True))
        group_parts.append(sub['experiment_id'].str.replace(r'__every\d+$', '', regex=True).reset_index(drop=True))
        steps.extend([k] * len(sub))
    return (pd.concat(x_parts, ignore_index=True), pd.concat(e_parts, ignore_index=True),
            pd.concat(y_parts, ignore_index=True), pd.concat(group_parts, ignore_index=True),
            np.asarray(steps))


def merge(x, e, mode, pca=None):
    if mode == 'baseline':
        return x
    a = e.to_numpy(dtype=np.float32, copy=False)
    if pca is not None:
        a = pca.transform(a).astype(np.float32)
    cols = [f'dino_{mode}_{i:03d}' for i in range(a.shape[1])]
    return pd.concat([x.reset_index(drop=True), pd.DataFrame(a, columns=cols)], axis=1)


def make_pca(mode, embeddings):
    if mode not in ('pca64', 'pca128'):
        return None
    n = int(mode[3:])
    if min(embeddings.shape) < n:
        raise ValueError(f'PCA {n}: 학습 행 수가 충분하지 않습니다.')
    return PCA(n_components=n, svd_solver='randomized', random_state=42).fit(embeddings)


def dump_report(name, y, pred, output):
    info = {'macro_f1': score(y, pred), 'per_class_f1': class_scores(y, pred),
            'confusion_matrix': confusion_matrix(y, pred, labels=LABELS).tolist()}
    print(name, 'Macro F1:', round(info['macro_f1'], 5),
          '| 4:', round(info['per_class_f1']['4'], 5), '| 5:', round(info['per_class_f1']['5'], 5), flush=True)
    output[name] = info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--baseline_model', type=Path, required=True)
    ap.add_argument('--cache_dir', type=Path, default=Path('cache_v4'))
    ap.add_argument('--output_dir', type=Path, default=Path('outputs_v4'))
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--image_size', type=int, default=224)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--n_splits', type=int, default=5)
    ap.add_argument('--augment_steps', type=int, nargs='*', default=[2, 3])
    ap.add_argument('--modes', nargs='+', choices=['baseline', 'pca64', 'pca128', 'full'],
                    default=['baseline', 'pca64', 'pca128', 'full'])
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = joblib.load(args.baseline_model)
    assert len(baseline['features']) == 190 and not baseline['spec']['interactions'], \
        '190개 특징 B 모델(model_v3_noix.joblib)이 필요합니다.'
    spec = baseline['spec']
    train, val = (read_split(args.data_dir, x) for x in ('train', 'validation'))
    assert set(train.experiment_id).isdisjoint(val.experiment_id), 'Train/Validation 실험 ID 겹침!'
    assert train.sample_id.is_unique and val.sample_id.is_unique

    mats = {}
    for split, frame in [('train', train), ('validation', val)]:
        paths = M.resolve_image_paths(frame, [args.data_dir / split, args.data_dir])
        stat_cache = args.cache_dir / f'img_features_v2_{split}.csv'
        if stat_cache.exists():
            img = pd.read_csv(stat_cache, dtype={'sample_id': 'string'}, float_precision='round_trip').set_index('sample_id')
            if set(img.index) != set(frame.sample_id) or list(img.columns) != M.IMAGE_STATS:
                raise ValueError(f'잘못된 이미지 통계 캐시: {stat_cache} (지우고 다시 생성하세요)')
        else:
            img = M.extract_image_features(frame, paths)
            stat_cache.parent.mkdir(parents=True, exist_ok=True)
            img.to_csv(stat_cache)
        emb = extract_embeddings(frame, paths, args.cache_dir, split,
                                 args.batch_size, args.image_size, args.device)
        x, e, y, g, steps = features_by_split(frame, img, spec, emb,
                                               args.augment_steps if split == 'train' else [])
        x = x[baseline['features']]
        assert len(x.columns) == 190
        mats[split] = (x, e, y, g, steps, img, emb)

    xtr, etr, ytr, gtr, steps, _, _ = mats['train']
    xv, ev, yv, _, _, imgv, embv = mats['validation']
    # Subsampling must RECOMPUTE time-window features, not merely take every third X row.
    val15 = M.subsample_experiments(val, 3)
    xv15 = M.build_features(val15, imgv, spec)[baseline['features']].reset_index(drop=True)
    ev15 = embv.loc[val15.sample_id.astype(str).to_numpy()].reset_index(drop=True)
    yv15 = pd.to_numeric(val15.label).astype(int).to_numpy()

    # Reproduce exact baseline model settings to isolate the contribution of DINO embeddings.
    def fresh_model():
        return M.build_model(baseline['class_weight'], 'lgbm')

    result = {'baseline_feature_count': 190, 'dinov2_dimension': DIM,
              'note': 'CV on original train rows only; groups by experiment; PCA fit within train fold; no postprocess',
              'experiments': {}}
    orig = (steps == 1)
    gkf = list(GroupKFold(n_splits=args.n_splits).split(xtr, ytr, groups=gtr))
    for mode in args.modes:
        print('\n==========', mode, '==========', flush=True)
        oof_pred = np.zeros(len(train), dtype=int)
        fold_scores = []
        for fold, (ti, vi) in enumerate(gkf, 1):
            orig_train = ti[orig[ti]]
            fold_pca = make_pca(mode, etr.iloc[orig_train].to_numpy())
            xt = merge(xtr.iloc[ti], etr.iloc[ti], mode, fold_pca)
            vi_orig = vi[orig[vi]]
            x_eval = merge(xtr.iloc[vi_orig], etr.iloc[vi_orig], mode, fold_pca)
            m = fresh_model().fit(xt, ytr.iloc[ti])
            pred = m.predict(x_eval)
            oof_pred[vi_orig] = pred
            fold_scores.append(score(ytr.iloc[vi_orig], pred))
            print(f'fold {fold}: {fold_scores[-1]:.5f}', flush=True)
        if (oof_pred == 0).any():
            raise AssertionError('OOF 일부 행이 예측되지 않았습니다.')
        report = {'cv_fold_scores': fold_scores, 'cv_mean': float(np.mean(fold_scores)),
                  'cv_std': float(np.std(fold_scores)), 'cv_oof': {}}
        dump_report('train_oof', ytr.iloc[np.where(orig)[0]], oof_pred, report['cv_oof'])
        pd.DataFrame({'sample_id': train.sample_id, 'label': ytr.iloc[np.where(orig)[0]].values,
                      'pred': oof_pred}).to_csv(args.output_dir / f'oof_v4_{mode}.csv', index=False)

        pca_final = make_pca(mode, etr.iloc[np.where(orig)[0]].to_numpy())
        model = fresh_model().fit(merge(xtr, etr, mode, pca_final), ytr)
        out = {}
        dump_report('validation_5s', yv, model.predict(merge(xv, ev, mode, pca_final)), out)
        dump_report('validation_15s', yv15, model.predict(merge(xv15, ev15, mode, pca_final)), out)
        report.update(out)
        result['experiments'][mode] = report
        joblib.dump({'model': model, 'spec': spec, 'features': list(baseline['features']),
                     'mode': mode, 'pca': pca_final, 'postprocess': None,
                     'image_size': args.image_size, 'dinov2_model': 'dinov2_vits14',
                     'class_weight': baseline['class_weight']}, args.output_dir / f'model_v4_{mode}.joblib')
        with open(args.output_dir / 'comparison_v4.json', 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print('\n비교 요약 (Validation은 최종 확인용):')
    for k, r in result['experiments'].items():
        print(k, 'CV=', round(r['cv_mean'], 5),
              'Val=', round(r['validation_5s']['macro_f1'], 5),
              'Val15=', round(r['validation_15s']['macro_f1'], 5),
              'F1-4=', round(r['validation_5s']['per_class_f1']['4'], 5),
              'F1-5=', round(r['validation_5s']['per_class_f1']['5'], 5))


if __name__ == '__main__':
    main()
