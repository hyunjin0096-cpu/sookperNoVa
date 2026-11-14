"""Evaluate saved v4 DINOv2+LightGBM models without retraining.

Example:
 python evaluate_v4.py --data_dir /home/ubuntu/data/battery_data_light \
     --model outputs_v4/model_v4_pca64.joblib

Use --models_dir outputs_v4 to evaluate all saved modes at once.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import model as M
from dino_features import extract_embeddings
from train_v4 import merge


def get_models(args):
    if args.model is not None:
        return [args.model]
    paths = [args.models_dir / f'model_v4_{mode}.joblib'
             for mode in ('baseline', 'pca64', 'pca128', 'full')]
    models = [p for p in paths if p.is_file()]
    if not models:
        raise FileNotFoundError(f'{args.models_dir}에서 model_v4_*.joblib을 찾지 못했습니다. 먼저 train_v4.py를 실행하세요.')
    return models


def report(y, pred):
    labels = M.LABELS
    fs = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
        'per_class_f1': {str(k): float(v) for k, v in zip(labels, fs)},
        'confusion_matrix': cm.tolist(),
        'actual_5_predicted_4': int(cm[labels.index(5), labels.index(4)]),
        'support': {str(k): int((np.asarray(y) == k).sum()) for k in labels},
    }


def main():
    parser = argparse.ArgumentParser(description='저장된 v4 모델 평가 (재학습 없음)')
    parser.add_argument('--data_dir', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--model', type=Path, help='하나의 v4 joblib 파일')
    group.add_argument('--models_dir', type=Path, help='v4 모델 전체가 저장된 폴더')
    parser.add_argument('--split', default='validation', help='라벨이 있는 split (기본 validation)')
    parser.add_argument('--cache_dir', type=Path, default=Path('cache_v4'))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', type=Path, default=Path('outputs_v4/evaluation_v4.json'))
    args = parser.parse_args()
    if args.models_dir is None and args.model is None:
        args.models_dir = Path('outputs_v4')

    frame_path = args.data_dir / args.split / f'{args.split}.csv'
    if not frame_path.is_file():
        raise FileNotFoundError(f'평가 CSV가 없습니다: {frame_path}')
    frame = pd.read_csv(frame_path, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    if 'label' not in frame:
        raise ValueError('평가용 CSV에 label 컬럼이 필요합니다.')
    if not frame['sample_id'].is_unique:
        raise ValueError('sample_id가 중복되었습니다.')
    y = pd.to_numeric(frame['label'], errors='raise').astype(int).to_numpy()
    sub15 = M.subsample_experiments(frame, 3)
    y15 = pd.to_numeric(sub15['label'], errors='raise').astype(int).to_numpy()

    models = [(path, joblib.load(path)) for path in get_models(args)]
    for path, bundle in models:
        if not {'model', 'spec', 'features', 'mode', 'pca', 'image_size'}.issubset(bundle):
            raise ValueError(f'{path}: v4 모델 형식이 아닙니다. train_v4.py로 생성한 모델을 사용하세요.')
        if bundle['mode'] != 'baseline' and bundle.get('dinov2_model') != 'dinov2_vits14':
            raise ValueError(f'{path}: DINOv2 모델 종류가 다릅니다.')

    paths = M.resolve_image_paths(frame, [args.data_dir / args.split, args.data_dir])
    stat_cache = args.cache_dir / f'img_features_v2_{args.split}.csv'
    if stat_cache.exists():
        img = pd.read_csv(stat_cache, dtype={'sample_id': 'string'}, float_precision='round_trip').set_index('sample_id')
        if set(img.index.astype(str)) != set(frame.sample_id.astype(str)) or list(img.columns) != M.IMAGE_STATS:
            raise ValueError(f'이미지 통계 캐시가 데이터와 다릅니다: {stat_cache}. 확인 후 삭제하고 재실행하세요.')
        print('기존 이미지 통계 캐시 사용:', stat_cache, flush=True)
    else:
        img = M.extract_image_features(frame, paths)
        stat_cache.parent.mkdir(parents=True, exist_ok=True)
        img.to_csv(stat_cache)
        print('이미지 통계 캐시 저장:', stat_cache, flush=True)

    # Embedding extraction is expensive; share by image size across models.
    emb_by_size = {}
    result = {}
    for path, bundle in models:
        mode = bundle['mode']
        features = bundle['features']
        x = M.build_features(frame, img, bundle['spec'])[features].reset_index(drop=True)
        x15 = M.build_features(sub15, img, bundle['spec'])[features].reset_index(drop=True)
        if mode == 'baseline':
            x_eval, x15_eval = x, x15
        else:
            size = bundle['image_size']
            if size not in emb_by_size:
                emb_by_size[size] = extract_embeddings(frame, paths, args.cache_dir, args.split,
                                                       args.batch_size, size, args.device)
            emb = emb_by_size[size]
            e = emb.loc[frame.sample_id.astype(str).to_numpy()].reset_index(drop=True)
            e15 = emb.loc[sub15.sample_id.astype(str).to_numpy()].reset_index(drop=True)
            x_eval = merge(x, e, mode, bundle['pca'])
            x15_eval = merge(x15, e15, mode, bundle['pca'])
        pred = np.asarray(bundle['model'].predict(x_eval), dtype=int).reshape(-1)
        pred15 = np.asarray(bundle['model'].predict(x15_eval), dtype=int).reshape(-1)
        stats = {'validation_5s': report(y, pred), 'validation_15s': report(y15, pred15)}
        result[mode] = stats
        print(f'\n====== {mode}: {path} ======')
        for label, yy, pp in [('Validation (원본 간격)', y, pred), ('Validation (15초 간격)', y15, pred15)]:
            r = report(yy, pp)
            print(f"{label}: Macro F1={r['macro_f1']:.4f} | F1-4={r['per_class_f1']['4']:.4f} "
                  f"| F1-5={r['per_class_f1']['5']:.4f} | 실제5→예측4={r['actual_5_predicted_4']}")
            print(classification_report(yy, pp, labels=M.LABELS, digits=4, zero_division=0))
            print(pd.DataFrame(r['confusion_matrix'], index=[f'실제{k}' for k in M.LABELS],
                               columns=[f'예측{k}' for k in M.LABELS]).to_string())
    if 'baseline' in result:
        base = result['baseline']['validation_5s']
        print('\n====== 기준 모델 대비 변화 (Validation 원본 간격) ======')
        for mode, stats in result.items():
            r = stats['validation_5s']
            print(f"{mode}: ΔMacro F1={r['macro_f1']-base['macro_f1']:+.4f}, "
                  f"ΔF1-4={r['per_class_f1']['4']-base['per_class_f1']['4']:+.4f}, "
                  f"ΔF1-5={r['per_class_f1']['5']-base['per_class_f1']['5']:+.4f}, "
                  f"Δ(실제5→예측4)={r['actual_5_predicted_4']-base['actual_5_predicted_4']:+d}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n평가 결과 저장:', args.output)


if __name__ == '__main__':
    main()
