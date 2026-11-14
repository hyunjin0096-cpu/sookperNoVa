"""제출 전 자가 점검: 실제 test.py 와 같은 함수로 B(미래 잘라내기)·C(실험 분리)·행 섞기·D(빈 이미지)를 검사합니다.

예) python selfcheck.py --data_dir ./work/data/battery_data_light --model ./model.joblib
통과 기준: B, C, 행 섞기 불일치 0.00%  /  D 예측 변화율 0% 초과
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from PIL import Image

import model as M

p = argparse.ArgumentParser()
p.add_argument('--data_dir', type=Path, required=True)
p.add_argument('--model', type=Path, default=Path('model.joblib'))
a = p.parse_args()
roots = [a.data_dir / 'test', a.data_dir]
test = pd.read_csv(a.data_dir / 'test' / 'test.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
bundle = joblib.load(a.model)


def predict(frame):
    paths = M.resolve_image_paths(frame, roots)
    img = M.extract_image_features(frame, paths)
    return pd.Series(M.predict_labels(bundle, frame, M.build_features(frame, img)), index=frame.sample_id.tolist())


base = predict(test)
diff = lambda q: float((base.loc[q.index] != q).mean())
half = test[test.relative_time_s <= test.groupby('experiment_id').relative_time_s.transform('median')]
exps = sorted(test.experiment_id.unique())
split = pd.concat([predict(test[test.experiment_id.isin(exps[0::2])]), predict(test[test.experiment_id.isin(exps[1::2])])])
shuffled = predict(test.sample(frac=1, random_state=1))
first = M.resolve_image_paths(test.head(1), roots).iloc[0]
blank = Path('blank_check.png').resolve()
with Image.open(first) as im:
    Image.new(im.mode, im.size).save(blank)
blank_rate = diff(predict(test.assign(image_path=str(blank))))

res = {'B 미래 잘라내기': diff(predict(half)), 'C 실험 분리': diff(split), '행 섞기': diff(shuffled)}
ok = True
for k, v in res.items():
    print(f'{k}: 불일치 {v:.2%}', '통과' if v == 0 else '실패')
    ok &= v == 0
print(f'D 빈 이미지: 예측 변화 {blank_rate:.2%}', '통과' if blank_rate > 0 else '실패 (이미지를 예측에 쓰지 않음)')
ok &= blank_rate > 0
print('\n전체:', '통과' if ok else '실패')
