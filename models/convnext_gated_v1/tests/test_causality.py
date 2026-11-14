"""Local invariant tests; safe to run without GPU/image data."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import build_sensor_features, augment_cadences


def sample():
    rows = []
    for exp, meth in [("E_A", 1), ("E_B", 2)]:
        for idx, t in enumerate([0, 5, 15, 20, 35, 50, 65, 95]):
            rows.append({
                "sample_id": f"{exp}-{idx}", "experiment_id": exp,
                "relative_time_s": t, "image_path": f"images/{exp}-{idx}.png",
                "meas_v": 5.0 - 0.1*idx, "meas_t1": 30 + min(idx, 3)*5 - max(0, idx-3)*3,
                "meas_t2": 33 + min(idx, 4)*4 - max(0, idx-4)*5,
                "meas_p": 1.0 + idx*0.04, "thermal_t": 100 + idx,
                "atmp_t": 23, "indu_t": 25.0 if meth==1 else np.nan,
                "bat_cc": 6, "bat_class": 1, "soc_status": 2, "tr_method": meth,
                **{g: 0.0 if idx % 2 else float(idx) for g in
                ("meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl",
                 "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2")},
            })
    return pd.DataFrame(rows)


def assert_equal_rows(a, b, ids_a, ids_b):
    a = a.copy(); b = b.copy()
    a.index = ids_a; b.index = ids_b
    common = a.index.intersection(b.index)
    assert len(common) and a.loc[common].equals(b.loc[common]), "Current/past feature drift"


def main():
    f = sample()
    full = build_sensor_features(f)
    shuffled = f.sample(frac=1, random_state=2026).reset_index(drop=True)
    assert_equal_rows(full, build_sensor_features(shuffled), f.sample_id, shuffled.sample_id)
    sub = f.groupby('experiment_id', sort=False).head(5).reset_index(drop=True)
    assert_equal_rows(full, build_sensor_features(sub), f.sample_id, sub.sample_id)
    one = f[f.experiment_id == 'E_A'].reset_index(drop=True)
    assert_equal_rows(full, build_sensor_features(one), f.sample_id, one.sample_id)
    assert full.loc[7, 'meas_t1_drawdown'] > 0, "Cooling-stage peak decline not captured"
    assert full.loc[7, 'meas_t1_slope_prev'] < 0, "Cooling-stage negative slope missing"
    assert full.loc[7, 'elapsed_since_previous'] == 30, "Actual time delta must be used"
    assert full.loc[f.tr_method==2, 'indu_t_missing'].eq(1).all()
    x_aug, src = augment_cadences(f)
    assert len(x_aug) > len(f) and len(src) == len(x_aug)
    assert list(full.columns) == list(x_aug.columns)
    print('PASS: reorder, remove future, split experiments, cooling drawdown, time gaps, method missingness, cadence views')

if __name__ == '__main__': main()
