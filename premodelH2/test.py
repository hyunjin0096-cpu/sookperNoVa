"""추론 스크립트 (재학습 없이 결과 재현).

예) python test.py --input /home/ubuntu/data/battery_data_light/test/test.csv \
                   --image_root /home/ubuntu/data/battery_data_light/test \
                   --model model.joblib --output submission.csv \
                   --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv

입력 CSV 에 들어 있는 행만으로 이미지 피처·CNN 확률·시계열 피처를 계산하고, 같은 실험의 과거·현재 관측만 씁니다.
CNN 은 기본 CPU 에서 이미지 1장씩 계산하므로 Test 일부만 넣거나 실험을 나누어도 같은 샘플은 같은 예측이 나옵니다.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

import model as M


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--image_root', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('model.joblib'))
    p.add_argument('--output', type=Path, default=Path('submission.csv'))
    p.add_argument('--sample_submission', type=Path, default=None, help='주면 제출 양식의 행 순서로 저장')
    p.add_argument('--device', default='cpu', help='CNN 추론 장치 (재현성을 위해 기본 cpu)')
    args = p.parse_args()

    frame = pd.read_csv(args.input, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    assert frame.sample_id.is_unique
    bundle = joblib.load(args.model)
    pred = M.predict_frame(frame, [args.image_root, args.image_root.parent], bundle, args.model, args.device)

    if args.sample_submission is not None:
        order = pd.read_csv(args.sample_submission, dtype={'sample_id': 'string'})['sample_id'].astype(str)
        assert set(order) == set(pred.index), '입력 CSV와 제출 양식의 ID가 다릅니다.'
        ids = order.tolist()
    else:
        ids = frame.sample_id.astype(str).tolist()
    submission = pd.DataFrame({'sample_id': ids, 'label': pred.loc[ids].values.astype(int)})
    assert submission.label.isin(M.LABELS).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f'저장: {args.output} ({len(submission)}행)')
    print(submission.label.value_counts().sort_index().to_string())


if __name__ == '__main__':
    main()
