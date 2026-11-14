"""현재 및 과거 관측만 사용하는 피처. 원본 experiment_id/sample_id/label은 모델 입력으로 넣지 않음."""
import numpy as np
import pandas as pd
from baseline_features import build_features, add_causal_features
from config import EXTRA_NUMERIC_COLUMNS, GAS_PREFIXES


def select_extra_columns(train):
    gas = [c for c in train.columns if c.lower().startswith(GAS_PREFIXES)]
    selected = list(dict.fromkeys([c for c in EXTRA_NUMERIC_COLUMNS + gas if c in train.columns]))
    return [c for c in selected if c not in {'label','sample_id','experiment_id','image_path','relative_time_s'}]


def make_tabular(frame, image_stats, extra_columns):
    out = build_features(frame, image_stats).reset_index(drop=True)
    additional = pd.DataFrame(index=frame.index)
    for col in extra_columns:
        additional[col] = pd.to_numeric(frame[col], errors='coerce') if col in frame else np.nan
    additional = additional.reset_index(drop=True).replace([np.inf, -np.inf], np.nan)
    out = pd.concat([out, additional], axis=1)

    t1 = pd.to_numeric(frame['meas_t1'], errors='coerce')
    t2 = pd.to_numeric(frame['meas_t2'], errors='coerce')
    out['temp_difference'] = (t1 - t2).to_numpy()
    out['temp_average'] = ((t1 + t2) / 2).to_numpy()
    out['temp_pressure_product'] = (t1 * pd.to_numeric(frame['meas_p'], errors='coerce')).to_numpy()

    gas_columns = [c for c in extra_columns if c.lower().startswith(GAS_PREFIXES)]
    if gas_columns:
        gas = additional[gas_columns]
        out['gas_zero_count'] = gas.eq(0).sum(axis=1)
        out['gas_missing_count'] = gas.isna().sum(axis=1)
        out['gas_all_zero'] = gas.notna().all(axis=1).astype(int) * gas.eq(0).all(axis=1).astype(int)

    # 측정 시점 이전 10초/30초 평균·최대·델타 (기존 baseline과 동일한 인과 방향)
    # 시간 간격이 다르기 때문에 row 단위 rolling은 사용하지 않습니다.
    time_base = frame[['experiment_id','relative_time_s','meas_v','meas_t1','meas_t2','meas_p']].copy()
    time_base['relative_time_s'] = pd.to_numeric(time_base['relative_time_s'], errors='coerce')
    for s in ['meas_v','meas_t1','meas_t2','meas_p']:
        time_base[s] = pd.to_numeric(time_base[s], errors='coerce')
    if 'thermal_t' in frame:
        time_base['thermal_t'] = pd.to_numeric(frame['thermal_t'], errors='coerce')
    signals = [c for c in time_base.columns if c not in {'experiment_id', 'relative_time_s'}]
    out = pd.concat([out, add_causal_features(time_base, signals, window_s=10).reset_index(drop=True)], axis=1)
    if 'thermal_t' in time_base:
        out = pd.concat([out, add_causal_features(time_base, ['thermal_t'], window_s=30).reset_index(drop=True)], axis=1)
    return out.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def fuse(frame, img_embedding, image_stats, extra_columns, use_embeddings=True):
    tabular = make_tabular(frame, image_stats, extra_columns)
    if use_embeddings:
        return pd.concat([tabular, img_embedding.reset_index(drop=True)], axis=1)
    return tabular
