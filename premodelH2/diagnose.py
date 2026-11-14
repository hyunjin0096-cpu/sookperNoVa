"""어떤 실험에서 틀리는지 진단합니다 (학습된 모델은 건드리지 않음).

예) python diagnose.py --data_dir /home/ubuntu/data/battery_data_light --cache_dir cache
train.py 가 만든 cache 를 재사용하므로 이미지는 다시 계산하지 않습니다. 5~10분 걸립니다.
결과: 실험 조건(배터리 형태·용량·유도 방식·SOC)별 정확도, 틀리는 방향, 가장 못 맞추는 실험 15개.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import model as M
import train as T

p = argparse.ArgumentParser()
p.add_argument('--data_dir', type=Path, required=True)
p.add_argument('--cache_dir', type=Path, default=Path('cache'))
p.add_argument('--n_estimators', type=int, default=200)
a = p.parse_args()

tr, tr_img = T.load_split(a.data_dir, 'train', a.cache_dir)
va, va_img = T.load_split(a.data_dir, 'validation', a.cache_dir)
frame = pd.concat([tr, va], ignore_index=True)
img = pd.concat([tr_img, va_img])
y_by_id = pd.Series(frame['label'].astype(int).values, index=frame.sample_id.astype(str).values)
X, y, g, _ = T.build_augmented(frame, img, y_by_id, (1,))      # 행 순서 = frame 순서

pred = np.zeros(len(X), dtype=int)
for tr_idx, va_idx in GroupKFold(n_splits=5).split(X, y, groups=g):
    m = M._lgbm('multiclass', 'balanced', a.n_estimators, M.SEED).fit(X.iloc[tr_idx], y[tr_idx])
    pred[va_idx] = m.predict(X.iloc[va_idx])
print(f'\n전체 OOF Macro F1 {f1_score(y, pred, labels=M.LABELS, average="macro", zero_division=0):.4f}')

d = pd.DataFrame({'exp': g, 'y': y, 'pred': pred})
for c in [c for c in ['tr_method', 'bat_class', 'bat_cc', 'soc_status'] if c in frame.columns]:
    d[c] = frame[c].to_numpy()
d['ok'] = d.y == d.pred
d['low'] = d.pred < d.y
d['high'] = d.pred > d.y


def summarize(group):
    labels = sorted(set(group.y) | set(group.pred))
    err = (~group.ok).sum()
    return {
        '실험수': group['exp'].nunique(), '행수': len(group), '정확도': group.ok.mean(),
        'MacroF1': f1_score(group.y, group.pred, labels=labels, average='macro', zero_division=0),
        '틀린것중_낮게예측%': (group.low.sum() / err * 100) if err else np.nan,
    }


def table(by):
    return pd.DataFrame({k: summarize(grp) for k, grp in d.groupby(by)}).T


meta_cols = [c for c in ['tr_method', 'bat_class', 'bat_cc', 'soc_status'] if c in d.columns]
for c in meta_cols:
    print(f'\n[{c}별]')
    print(table(c).round(3).to_string())

per = table('exp')
if meta_cols:
    per = per.join(d.groupby('exp')[meta_cols].first())
print('\n[가장 못 맞추는 실험 15개] (정확도 낮은 순)')
print(per.sort_values('정확도').head(15).round(3).to_string())
acc = per['정확도'].astype(float)
print(f'\n실험별 정확도: 평균 {acc.mean():.3f} | 중앙값 {acc.median():.3f} | 0.8 미만 {int((acc < 0.8).sum())}개 / {len(per)}개')
