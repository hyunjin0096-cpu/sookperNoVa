"""v5 추론 스크립트 (모델 예측 + 단조성 보정, U-Net 모양 피처 선택).

예) python test.py --input ./data/battery_data_light/test/test.csv \
                   --image_root ./data/battery_data_light/test \
                   --model ./model.joblib --output ./submission.csv \
                   [--sample_submission ./data/battery_data_light/sample_submission.csv]

입력 CSV에 들어 있는 행만으로 피처를 만들기 때문에, 주최 측이 Test 일부만 넣거나
실험별로 나누어 실행해도 같은 샘플에는 같은 예측이 나와야 합니다.
입력 컬럼 규격(spec)은 학습 때 Train으로 정한 것을 모델 파일에서 그대로 불러 씁니다.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import model as M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--image_root', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('model.joblib'))
    parser.add_argument('--unet', type=Path, default=None, help='U-Net 가중치(모델이 U-Net 피처를 쓰면 필요. 생략 시 모델 파일 옆의 저장된 이름)')
    parser.add_argument('--output', type=Path, default=Path('submission.csv'))
    parser.add_argument('--sample_submission', type=Path, default=None, help='주면 제출 양식의 행 순서로 저장')
    parser.add_argument('--no_postprocess', action='store_true', help='단조성 보정 없이 모델 예측 그대로 저장')
    args = parser.parse_args()

    frame = pd.read_csv(args.input, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    assert frame.sample_id.is_unique
    bundle = joblib.load(args.model)
    paths = M.resolve_image_paths(frame, [args.image_root, args.image_root.parent])
    img = M.extract_image_features(frame, paths)
    if bundle['spec'].get('unet'):
        import unet as U
        weights = args.unet or args.model.parent / bundle['unet_weights']
        img = img.join(U.extract_unet_features(frame, paths, weights, device='cpu'))   # CPU 고정(재현성)
        print('U-Net 모양 피처 사용:', weights)
    X = M.build_features(frame, img, bundle['spec'])[bundle['features']]
    proba = bundle['model'].predict_proba(X)
    if bundle.get('postprocess') is not None and not args.no_postprocess:
        # 단조성 보정: 같은 실험의 과거~현재 확률만 사용(입력에 있는 행만으로 계산)
        proba = M.forward_filter(proba, frame['experiment_id'].to_numpy(),
                                 X['relative_time_s'].to_numpy(), bundle['postprocess'])
        print('단조성 보정 적용')
    pred = pd.Series(np.array(M.LABELS)[proba.argmax(axis=1)], index=frame.sample_id.values)

    if args.sample_submission is not None:
        order = pd.read_csv(args.sample_submission, dtype={'sample_id': 'string'})['sample_id']
        assert set(order) == set(pred.index), '입력 CSV와 제출 양식의 ID가 다릅니다.'
        ids = order.tolist()
    else:
        ids = frame.sample_id.tolist()
    submission = pd.DataFrame({'sample_id': ids, 'label': pred.loc[ids].values})
    assert submission.label.isin(M.LABELS).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f'저장: {args.output} ({len(submission)}행)')


if __name__ == '__main__':
    main()
