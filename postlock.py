"""제출 파일에 '단조 잠금' 후처리만 적용합니다 (어떤 모델의 제출 파일이든 사용 가능).

예) python postlock.py --test_csv /home/ubuntu/data/battery_data_light/test/test.csv \
                       --input submission_j1_csv.csv --output submission_j1_lock.csv --confirm_s 30

규칙: 같은 실험에서 confirm_s 초 이상 계속 s단계 이상으로 예측되면 s단계로 잠그고, 이후 그 아래로는 내리지 않습니다.
같은 실험의 현재·과거 예측만 사용하므로 미래 관측·다른 실험 정보를 쓰지 않습니다 (규정 준수).
최종 제출에 쓰려면 그 모델의 test.py 마지막에 apply_lock 을 붙여 코드로 재현되게 하십시오.
"""
import argparse

import numpy as np
import pandas as pd


def apply_lock(pred, experiment_id, t, confirm_s):
    pred = np.asarray(pred, dtype=int)
    t = np.asarray(t, dtype=float)
    out = pred.copy()
    for _, idx in pd.Series(np.asarray(experiment_id)).groupby(np.asarray(experiment_id), sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        lock = 1
        run_start = {s: None for s in range(2, 7)}
        for i in idx:
            p = pred[i]
            for s in range(2, 7):
                if p >= s:
                    if run_start[s] is None:
                        run_start[s] = t[i]
                else:
                    run_start[s] = None
            for s in range(6, 1, -1):
                if run_start[s] is not None and t[i] - run_start[s] >= confirm_s:
                    lock = max(lock, s)
                    break
            out[i] = max(p, lock)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test_csv', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--confirm_s', type=float, default=30)
    a = p.parse_args()
    test = pd.read_csv(a.test_csv, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    sub = pd.read_csv(a.input, dtype={'sample_id': 'string'})
    lab = sub.set_index('sample_id').label.reindex(test.sample_id)
    assert lab.notna().all(), '제출 파일과 test.csv 의 sample_id 가 다릅니다.'
    new = apply_lock(lab.values, test.experiment_id.values, test.relative_time_s.values, a.confirm_s)
    out = pd.Series(new, index=test.sample_id.values).reindex(sub.sample_id)
    changed = int((out.values != sub.label.values).sum())
    pd.DataFrame({'sample_id': sub.sample_id, 'label': out.values.astype(int)}).to_csv(a.output, index=False)
    print(f'저장: {a.output} | 바뀐 행 {changed}개 ({changed / len(sub):.2%})')
    print(pd.crosstab(sub.label.values, out.values, rownames=['원래'], colnames=['잠금 후']).to_string())


if __name__ == '__main__':
    main()