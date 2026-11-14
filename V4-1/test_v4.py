"""v4 inference. Example:
python test_v4.py --input data/battery_data_light/test/test.csv --image_root data/battery_data_light/test --model outputs_v4/model_v4_pca64.joblib --output submission_v4.csv --sample_submission data/battery_data_light/sample_submission.csv
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import model as M
from dino_features import extract_embeddings
from train_v4 import merge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, type=Path)
    ap.add_argument('--image_root', required=True, type=Path)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--output', type=Path, default=Path('submission_v4.csv'))
    ap.add_argument('--sample_submission', type=Path)
    ap.add_argument('--cache_dir', type=Path, default=Path('cache_v4'))
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    frame = pd.read_csv(args.input, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    assert frame.sample_id.is_unique, 'sample_id 중복'
    bundle = joblib.load(args.model)
    paths = M.resolve_image_paths(frame, [args.image_root, args.image_root.parent])
    img = M.extract_image_features(frame, paths)
    x = M.build_features(frame, img, bundle['spec'])[bundle['features']].reset_index(drop=True)
    mode = bundle['mode']
    if mode == 'baseline':
        x_input = x
    else:
        emb = extract_embeddings(frame, paths, args.cache_dir,
                                 'test', args.batch_size, bundle['image_size'], args.device)
        e = emb.loc[frame.sample_id.astype(str).to_numpy()].reset_index(drop=True)
        x_input = merge(x, e, mode, bundle['pca'])
    pred = np.asarray(bundle['model'].predict(x_input), dtype=int).reshape(-1)
    ids = frame.sample_id.astype(str).tolist()
    output = pd.DataFrame({'sample_id': ids, 'label': pred})
    assert output.label.isin(M.LABELS).all(), '예측 범위 확인'
    if args.sample_submission:
        order = pd.read_csv(args.sample_submission, dtype={'sample_id': 'string'}).sample_id
        assert len(order) == len(output) and set(order) == set(ids), 'sample_submission의 ID가 일치하지 않습니다.'
        output = output.set_index('sample_id').loc[order.astype(str).tolist()].reset_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f'저장 완료: {args.output} ({len(output):,} rows)')


if __name__ == '__main__':
    main()
