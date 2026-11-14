"""Test CSV 전체 또는 일부만 전달되어도 같은 행에 같은 결과를 반환합니다.
예: python test.py --input data/battery_data_light/test/test.csv --image_root data/battery_data_light --output submission.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import WEIGHTS_DIR, DATA_DIR
from data_utils import read_split, image_roots
from features import fuse
from image_encoder import load_encoder, extract


def predict_frame(frame, model, meta, encoder, device, roots, blank=False):
    embeddings, stats = extract(frame, roots, encoder, device, blank=blank)
    X = fuse(frame, embeddings, stats, meta['extra_columns'], use_embeddings=not meta['tabular_only'])
    X = X.reindex(columns=meta['features'])
    pred = model.predict(X).astype(int) + 1
    assert np.isin(pred, np.arange(1,7)).all()
    return pd.Series(pred, index=frame.sample_id.astype(str).to_list(), name='label')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=None, help='평가용 Test CSV. 없으면 기본 test/test.csv')
    ap.add_argument('--output', default='submission.csv')
    ap.add_argument('--image_root', required=True, help='CSV image_path의 기준 이미지 폴더')
    ap.add_argument('--weights', default=None, help='생략 시 weights/resnet18.pth 사용')
    ap.add_argument('--check', action='store_true', help='미래·실험 분리·행 순서·빈 이미지 검사')
    args = ap.parse_args()
    meta = json.loads((WEIGHTS_DIR / 'metadata.json').read_text(encoding='utf-8'))
    model = joblib.load(WEIGHTS_DIR / 'lightgbm.joblib')
    encoder_weights = Path(args.weights) if args.weights else WEIGHTS_DIR / 'resnet18.pth'
    if not encoder_weights.is_file():
        raise FileNotFoundError(f'{encoder_weights} 가 없습니다. README의 기존 가중치 이전 절차를 확인하세요.')
    encoder, device = load_encoder(str(encoder_weights))
    frame = pd.read_csv(args.input, dtype={'sample_id':'string','experiment_id':'string'}) if args.input else read_split('test')
    image_root = Path(args.image_root).expanduser().resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f'이미지 폴더가 없습니다: {image_root}')
    # csv image_path가 split 상대 경로이거나 이미지 루트 상대 경로일 수 있음
    csv_root = Path(args.input).resolve().parent if args.input else DATA_DIR / 'test'
    roots = list(dict.fromkeys([image_root, image_root / 'test', csv_root, csv_root.parent]))
    base = predict_frame(frame,model,meta,encoder,device,roots)
    if args.check:
        sample = frame.groupby('experiment_id', sort=False).head(25).copy()  # 빠른 검사용
        ref = predict_frame(sample,model,meta,encoder,device,roots)
        shuffled = predict_frame(sample.sample(frac=1, random_state=42),model,meta,encoder,device,roots)
        assert ref.sort_index().equals(shuffled.sort_index()), '행 순서 변경 검사 실패'
        half = sample[sample.relative_time_s <= sample.groupby('experiment_id').relative_time_s.transform('median')]
        half_pred = predict_frame(half,model,meta,encoder,device,roots)
        assert ref.loc[half_pred.index].equals(half_pred), '미래 제거 검사 실패'
        group_pred = pd.concat([predict_frame(chunk,model,meta,encoder,device,roots) for _,chunk in sample.groupby('experiment_id')])
        assert ref.sort_index().equals(group_pred.sort_index()), '실험 분리 검사 실패'
        blank_pred = predict_frame(sample,model,meta,encoder,device,roots,blank=True)
        changed = (ref != blank_pred).mean()
        print(f'빈 이미지 변경 시 클래스 변화율(간이 검사): {changed:.2%}')
        if changed == 0:
            print('경고: 일부 샘플에서 이미지 의존성이 확인되지 않았습니다. 전체 Test 및 공식 검증기로 추가 확인이 필요합니다.')
        print('기본 행 순서/미래/실험 분리 검사 통과')

    path = Path(args.output)
    submission_path = DATA_DIR / 'sample_submission.csv'
    if args.input is None and submission_path.exists():
        template = pd.read_csv(submission_path, dtype={'sample_id':'string'})
        if set(template.sample_id) != set(frame.sample_id):
            raise ValueError('Test와 sample_submission ID 불일치')
        result = template[['sample_id']].copy()
        result['label'] = result.sample_id.astype(str).map(base)
    else:
        result = frame[['sample_id']].copy()
        result['label'] = result.sample_id.astype(str).map(base)
    assert result.label.notna().all()
    result['label'] = result.label.astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path,index=False)
    print('제출 파일 저장:',path.resolve(),'행 수:',len(result))

if __name__ == '__main__':
    main()
