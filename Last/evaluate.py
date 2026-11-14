"""저장된 모델을 Validation으로 평가합니다 (재학습 없음).

예) python evaluate.py --data_dir ./data/battery_data_light --model model.joblib

주의: train.py를 --refit_train_val 로 학습한 모델은 Validation을 이미 학습했으므로 이 점수는 의미가 없습니다.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import model as M


def report(name, y, pred):
    print(f'\n===== {name} =====')
    print(f'Accuracy {accuracy_score(y, pred):.4f} | Macro F1 {f1_score(y, pred, labels=M.LABELS, average="macro", zero_division=0):.4f}')
    print(classification_report(y, pred, labels=M.LABELS, digits=4, zero_division=0))
    print(pd.DataFrame(confusion_matrix(y, pred, labels=M.LABELS),
                       index=[f'실제{l}' for l in M.LABELS], columns=[f'예측{l}' for l in M.LABELS]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('model.joblib'))
    parser.add_argument('--unet', type=Path, default=None, help='U-Net 가중치(모델이 U-Net 피처를 쓰면 필요. 생략 시 모델 파일 옆의 저장된 이름)')
    parser.add_argument('--split', default='validation')
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    if bundle.get('refit_train_val') and args.split == 'validation':
        print('경고: 이 모델은 Train+Validation으로 학습되어 Validation 점수가 부풀려집니다.')
    frame = pd.read_csv(args.data_dir / args.split / f'{args.split}.csv',
                        dtype={'sample_id': 'string', 'experiment_id': 'string'})
    paths = M.resolve_image_paths(frame, [args.data_dir / args.split, args.data_dir])
    img = M.extract_image_features(frame, paths)
    if bundle['spec'].get('unet'):
        import unet as U
        img = img.join(U.extract_unet_features(frame, paths, args.unet or args.model.parent / bundle['unet_weights']))
    y = pd.to_numeric(frame['label']).astype(int)
    print('모델:', bundle.get('algo', 'lgbm'), '| class_weight:', bundle['class_weight'], '| 피처 수:', len(bundle['features']))

    pp = bundle.get('postprocess')
    labels = np.array(M.LABELS)

    def predict(f):
        X = M.build_features(f, img, bundle['spec'])[bundle['features']]
        proba = bundle['model'].predict_proba(X)
        raw = labels[proba.argmax(axis=1)]
        post = None if pp is None else labels[M.forward_filter(
            proba, f['experiment_id'].to_numpy(), X['relative_time_s'].to_numpy(), pp).argmax(axis=1)]
        return raw, post

    raw, post = predict(frame)
    report(f'{args.split} (원본 간격) — 보정 전', y, raw)
    if post is not None:
        report(f'{args.split} (원본 간격) — 단조성 보정 후 (alpha={pp["alpha"]}, down_eps={pp["down_eps"]})', y, post)

    for step, sec in [(2, 10), (3, 15)]:
        sub = M.subsample_experiments(frame, step)
        ys = pd.to_numeric(sub['label']).astype(int)
        r, p_ = predict(sub)
        line = f'{args.split} {sec}초 간격: Macro F1 {f1_score(ys, r, labels=M.LABELS, average="macro", zero_division=0):.4f}'
        if p_ is not None:
            line += f' → 보정 후 {f1_score(ys, p_, labels=M.LABELS, average="macro", zero_division=0):.4f}'
        print(line)

    if 'tr_method' in frame.columns:
        final = post if post is not None else raw
        for v, idx in frame.groupby('tr_method').indices.items():
            f1 = f1_score(y.iloc[idx], final[idx], labels=M.LABELS, average='macro', zero_division=0)
            print(f'tr_method={v}: Macro F1 {f1:.4f} ({len(idx)}행)')

if __name__ == '__main__':
    main()
