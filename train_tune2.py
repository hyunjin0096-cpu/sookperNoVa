"""예: python train.py --weights ./resnet18-f37072fd.pth"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from config import ARTIFACT_DIR, CACHE_DIR, WEIGHTS_DIR, CV_FOLDS, SEED
from pathlib import Path
WEIGHTS_DIR = Path(__file__).resolve().parent / "tuning_num_leaves_7"

from data_utils import read_split, image_roots
from features import select_extra_columns, fuse
from image_encoder import load_encoder, extract, export_pretrained_weights

LABELS = list(range(6))



def new_model(n_estimators=1200):
    return LGBMClassifier(
        objective='multiclass',
        num_class=6,
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=7,
        min_child_samples=80,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=SEED,
        n_jobs=1,
        verbosity=-1,
    )

def get_features(split, frame, encoder, device, extra, no_cache=False, tabular_only=False):
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    cache = CACHE_DIR / f'images_{split}.joblib'
    if not no_cache and cache.exists():
        item = joblib.load(cache)
        if item['sample_ids'] == frame.sample_id.astype(str).tolist():
            print(f'{split}: 이미지 특징 캐시 사용')
            embed, stats = item['embeddings'], item['stats']
        else:
            raise ValueError(f'{cache}에 다른 sample_id가 들어 있습니다. --no-cache로 재생성하세요.')
    else:
        print(f'{split}: ResNet18 이미지 특징 생성')
        embed, stats = extract(frame, image_roots(split), encoder, device)
        joblib.dump({'sample_ids':frame.sample_id.astype(str).tolist(), 'embeddings':embed, 'stats':stats}, cache)
    return fuse(frame, embed, stats, extra, use_embeddings=not tabular_only)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=None, help='오프라인 ResNet18 사전학습 state_dict 경로')
    ap.add_argument('--no-cache', action='store_true', help='이미지 캐시를 새로 생성')
    ap.add_argument('--tabular-only', action='store_true', help='ResNet 512차원을 제외한 대조 실험')
    ap.add_argument('--folds', type=int, default=CV_FOLDS)
    args = ap.parse_args()
    WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)
    train = read_split('train')
    validation = read_split('validation')
    if 'label' not in train or 'label' not in validation:
        raise ValueError('train/validation에는 label 컬럼이 있어야 합니다.')
    assert set(train.experiment_id).isdisjoint(validation.experiment_id), 'Train/Validation 실험 ID 중복'
    y = train.label.astype(int).to_numpy() - 1
    y_val = validation.label.astype(int).to_numpy() - 1
    assert set(np.unique(y)).issubset(set(LABELS)) and set(np.unique(y_val)).issubset(set(LABELS))
    extra = select_extra_columns(train)
    print('추가 수치 컬럼:', extra)
    encoder, device = load_encoder(args.weights)
    print('ResNet18 실행 장치:', device)
    X = get_features('train', train, encoder, device, extra, args.no_cache, args.tabular_only)
    X_val = get_features('validation', validation, encoder, device, extra, args.no_cache, args.tabular_only)
    X_val = X_val.reindex(columns=X.columns)
    print('최종 특징 수:', len(X.columns))

    cv = GroupKFold(n_splits=args.folds)
    scores, best_rounds = [], []
    for i, (a,b) in enumerate(cv.split(X, y, groups=train.experiment_id), 1):
        clf = new_model()
        clf.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], eval_metric='multi_logloss',
                callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])
        pred = clf.predict(X.iloc[b])
        score = f1_score(y[b], pred, labels=LABELS, average='macro', zero_division=0)
        scores.append(float(score))
        best_rounds.append(int(clf.best_iteration_))
        print(f'Fold {i}: Macro F1={score:.4f}, trees={clf.best_iteration_}')
    print(f'CV Macro F1 {np.mean(scores):.4f} ± {np.std(scores):.4f}')
    
    trees = max(100, int(np.median(best_rounds)))
    clf = new_model(n_estimators=trees)
    clf.fit(X, y)
    pred = clf.predict(X_val)

    # 1. Train 데이터 성능
    train_pred = clf.predict(X)

    print("Train Macro F1:",
          f1_score(y, train_pred, labels=LABELS, average='macro', zero_division=0))

    # 2. Train과 Validation의 클래스 비율 비교
    print("\nTrain 클래스 비율")
    print(train['label'].value_counts(normalize=True).sort_index())

    print("\nValidation 클래스 비율")
    print(validation['label'].value_counts(normalize=True).sort_index())

    # 3. 실험 수 비교
    print("\nTrain 실험 수:", train['experiment_id'].nunique())
    print("Validation 실험 수:", validation['experiment_id'].nunique())

    # 4. Validation 클래스별 성능
    val_score = f1_score(
        y_val, pred, labels=LABELS, average='macro', zero_division=0
    )

    print(f'Validation Accuracy={accuracy_score(y_val, pred):.4f}')
    print(f'Validation Macro F1={val_score:.4f}')
    print(classification_report(
        y_val + 1, pred + 1, labels=list(range(1, 7)), zero_division=0
    ))
    print('Confusion matrix:\n',
          confusion_matrix(y_val + 1, pred + 1, labels=list(range(1, 7))))

    # 검증 데이터를 최종 재학습에 섞지 않습니다.
    joblib.dump(clf, WEIGHTS_DIR / 'lightgbm.joblib')

    meta = {
        'features': list(X.columns),
        'extra_columns': extra,
        'tabular_only': args.tabular_only,
        'trees': trees,
        'cv_macro_f1': scores,
        'validation_macro_f1': float(val_score),
        'weight_path_used_for_training': args.weights or 'torchvision default',
    }

    (WEIGHTS_DIR / 'metadata.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    if args.weights:
        import shutil
        if Path(args.weights).resolve() != (WEIGHTS_DIR / 'resnet18.pth').resolve():
            shutil.copy2(args.weights, WEIGHTS_DIR / 'resnet18.pth')
    else:
        export_pretrained_weights(WEIGHTS_DIR / 'resnet18.pth')

    print('저장 완료:', WEIGHTS_DIR.resolve())
    print('주의: --tabular-only로 학습한 결과는 대회 이미지 의존성 조건을 만족하지 못할 수 있습니다.')

if __name__ == '__main__':
    main()