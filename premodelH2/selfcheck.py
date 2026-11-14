"""제출 전 자가 점검 (test.py 와 같은 함수 사용).

예) python selfcheck.py --data_dir /home/ubuntu/data/battery_data_light --model model.joblib
통과 기준: B 미래 잘라내기·C 실험 분리·행 섞기 불일치 0.00%  /  D 빈 이미지 예측 변화 0% 초과
CNN 을 CPU 로 5번 계산하므로 10분 이상 걸릴 수 있습니다.
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
p.add_argument('--device', default='cpu')
a = p.parse_args()
roots = [a.data_dir / 'test', a.data_dir]
test = pd.read_csv(a.data_dir / 'test' / 'test.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
bundle = joblib.load(a.model)
predict = lambda frame: M.predict_frame(frame, roots, bundle, a.model, a.device)

print('기준 예측...')
base = predict(test)
diff = lambda q: float((base.loc[q.index] != q).mean())
half = test[test.relative_time_s <= test.groupby('experiment_id').relative_time_s.transform('median')]
exps = sorted(test.experiment_id.unique())
res = {}
print('B 미래 잘라내기...')
res['B 미래 잘라내기'] = diff(predict(half))
print('C 실험 분리...')
res['C 실험 분리'] = diff(pd.concat([predict(test[test.experiment_id.isin(exps[0::2])]),
                                   predict(test[test.experiment_id.isin(exps[1::2])])]))
print('행 섞기...')
res['행 섞기'] = diff(predict(test.sample(frac=1, random_state=1)))
print('D 빈 이미지...')
first = M.resolve_image_paths(test.head(1), roots).iloc[0]
blank = Path('blank_check.png').resolve()
with Image.open(first) as im:
    Image.new(im.mode, im.size).save(blank)
blank_rate = diff(predict(test.assign(image_path=str(blank))))

ok = True
print()
for k, v in res.items():
    print(f'{k}: 불일치 {v:.2%}', '통과' if v == 0 else '실패')
    ok &= v == 0
print(f'D 빈 이미지: 예측 변화 {blank_rate:.2%}', '통과' if blank_rate > 0 else '실패 (이미지를 예측에 쓰지 않음)')
ok &= blank_rate > 0
print('\n전체:', '통과' if ok else '실패')
