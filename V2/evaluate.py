"""저장된 모델을 Validation으로 평가합니다 (재학습 없음).

예) python evaluate.py --data_dir ./data/battery_data_light --model model.joblib

주의: train.py를 --refit_train_val 로 학습한 모델은 Validation을 이미 학습했으므로 이 점수는 의미가 없습니다.
"""
import argparse
from pathlib import Path

import joblib
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
    parser.add_argument('--split', default='validation')
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    if bundle.get('refit_train_val') and args.split == 'validation':
        print('경고: 이 모델은 Train+Validation으로 학습되어 Validation 점수가 부풀려집니다.')
    frame = pd.read_csv(args.data_dir / args.split / f'{args.split}.csv',
                        dtype={'sample_id': 'string', 'experiment_id': 'string'})
    img = M.extract_image_features(frame, M.resolve_image_paths(frame, [args.data_dir / args.split, args.data_dir]))
    y = pd.to_numeric(frame['label']).astype(int)
    print('class_weight:', bundle['class_weight'], '| 피처 수:', len(bundle['features']))

    X = M.build_features(frame, img, bundle['spec'])[bundle['features']]
    report(f'{args.split} (원본 간격)', y, bundle['model'].predict(X))

    for step, sec in [(2, 10), (3, 15)]:
        sub = M.subsample_experiments(frame, step)
        Xs = M.build_features(sub, img, bundle['spec'])[bundle['features']]
        ys = pd.to_numeric(sub['label']).astype(int)
        f1 = f1_score(ys, bundle['model'].predict(Xs), labels=M.LABELS, average='macro', zero_division=0)
        print(f'\n{args.split} {sec}초 간격으로 솎았을 때 Macro F1: {f1:.4f}')

    tr_method = frame.get('tr_method')
    if tr_method is not None:
        pred = pd.Series(bundle['model'].predict(X), index=frame.index)
        for v, idx in frame.groupby('tr_method').groups.items():
            f1 = f1_score(y[idx], pred[idx], labels=M.LABELS, average='macro', zero_division=0)
            print(f'tr_method={v}: Macro F1 {f1:.4f} ({len(idx)}행)')


if __name__ == '__main__':
    main()
