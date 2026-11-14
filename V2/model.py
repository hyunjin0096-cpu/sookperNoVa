"""Uni-D 배터리 열폭주 6단계 분류 — LightGBM + 확장 피처 (베이스라인 v7 기반).

베이스라인 대비 변경점
- 입력 컬럼: 센서 5개 → CSV의 숫자·범주 컬럼 자동 사용(thermal_t, 가스 센서, 실험 조건 등).
  meas_a(상수), indu_t.1(Test 전부 결측)은 제외합니다.
- 이미지: 흑백 밝기 통계 + 의사색(HSV) 기반 '열 지수' 통계. 이미지 크기는 실험 식별 지름길이 될 수 있어 쓰지 않습니다.
- 시계열: 30/60/120초 윈도우의 평균·최댓값·변화량·초당 기울기, 직전 관측 대비 기울기,
  현재까지의 누적 최댓값(전압은 최솟값), 첫 관측 대비 변화량. 모두 현재 시점 이하 관측만 사용합니다.
- 모델: 로지스틱 회귀 → LightGBM (결측값을 그대로 처리하므로 대체·표준화 없음).
train.py, test.py가 모두 이 모듈과 같은 코드를 사용합니다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from PIL import Image

SEED = 42
LABELS = list(range(1, 7))

# 피처로 쓰지 않는 컬럼
ID_COLS = ['sample_id', 'experiment_id', 'image_path', 'label', 'relative_time_s']
DROP_COLS = ['meas_a', 'indu_t.1']   # EDA: meas_a는 전 split 상수 0, indu_t.1은 Test 100% 결측
# 가스 센서 10개 (실제 CSV 컬럼명). CSV에 없는 이름은 자동으로 건너뜁니다.
GAS_COLS = ['meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl',
            'meas_hf', 'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']
MAX_CATEGORIES = 50                  # 숫자가 아닌 컬럼은 고유값이 이 이하일 때만 범주형으로 사용

# 시계열 피처 대상 신호 (CSV에 있는 것만 사용)
CORE_SENSORS = ['meas_v', 'meas_t1', 'meas_t2', 'meas_p', 'thermal_t']
IMAGE_WINDOW_SIGNALS = ['img_p99', 'heat_mean', 'heat_p99', 'heat_hot_ratio']
WINDOWS_S = (30.0, 60.0, 120.0)      # 과거 윈도우 길이(초). 행 수가 아니라 relative_time_s 기준
CUM_MIN_SIGNALS = {'meas_v'}         # 누적 최솟값을 쓰는 신호(전압은 떨어지는 방향이 의미 있음)

HOT_THRESHOLD = 0.8                  # 흑백 밝기 0~1 기준 고온 픽셀 임계값(고정값)


# ---------------------------------------------------------------------------
# 1. 입력 컬럼 규격 (Train에서 한 번 정하고 모델과 함께 저장 → Test에 그대로 적용)
# ---------------------------------------------------------------------------
def infer_feature_spec(train_frame: pd.DataFrame, exclude: list[str] | None = None) -> dict:
    skip = set(ID_COLS) | set(DROP_COLS) | set(exclude or [])
    numeric_cols, categorical, ignored = [], {}, []
    for c in train_frame.columns:
        if c in skip or 'label' in c.lower():
            continue
        raw = train_frame[c]
        conv = pd.to_numeric(raw, errors='coerce')
        if raw.notna().sum() == 0:
            ignored.append(c)
        elif conv.notna().sum() >= 0.95 * raw.notna().sum():
            if conv.nunique(dropna=True) > 1:
                numeric_cols.append(c)
            else:
                ignored.append(c)                       # 상수 컬럼
        elif raw.nunique(dropna=True) <= MAX_CATEGORIES:
            categorical[c] = sorted(raw.dropna().astype(str).unique().tolist())
        else:
            ignored.append(c)
    gas = [c for c in GAS_COLS if c in numeric_cols]
    if not gas:                                          # 이름이 다르면 'gas' 포함 컬럼으로 대체 탐지
        gas = [c for c in numeric_cols if 'gas' in c.lower()]
    sensors = [c for c in CORE_SENSORS if c in numeric_cols]
    window_signals = sensors + IMAGE_WINDOW_SIGNALS + (['gas_zero_count'] if gas else [])
    return {'numeric': numeric_cols, 'categorical': categorical, 'gas': gas,
            'window_signals': window_signals, 'ignored': ignored}


def describe_spec(spec: dict) -> str:
    return '\n'.join([
        f"  숫자 컬럼 {len(spec['numeric'])}개: {spec['numeric']}",
        f"  범주 컬럼 {len(spec['categorical'])}개: {list(spec['categorical'])}",
        f"  가스 센서 {len(spec['gas'])}개: {spec['gas']}",
        f"  시계열 대상 신호: {spec['window_signals']}",
        f"  제외(상수·전부 결측·고유값 과다): {spec['ignored']} / 고정 제외: {DROP_COLS}",
    ])


# ---------------------------------------------------------------------------
# 2. 이미지 경로 연결 (베이스라인과 동일)
# ---------------------------------------------------------------------------
def resolve_image_paths(frame: pd.DataFrame, roots: list[Path]) -> pd.Series:
    rel = frame['image_path'].astype(str)
    for root in roots:
        candidate = rel.map(lambda p: Path(root) / p)
        if candidate.map(Path.is_file).all():
            return candidate
    index = {}
    for root in roots:
        for path in Path(root).rglob('*'):
            if path.is_file():
                index.setdefault(path.name, path)
    candidate = rel.map(lambda p: index.get(Path(p).name))
    missing = candidate.isna().sum()
    assert missing == 0, f'이미지 {missing}개를 찾지 못했습니다. image_root를 확인하십시오. 예: {rel[candidate.isna()].iloc[0]}'
    return candidate


# ---------------------------------------------------------------------------
# 3. 이미지 피처 (샘플마다 독립 계산 → 다른 행 정보 사용 없음)
#    열화상은 의사색(보라→파랑→청록→초록→노랑→빨강→흰색 = 저온→고온)이라
#    흑백 변환만 하면 색상 정보가 사라집니다. 색상(Hue)을 고정 규칙으로 0~1 '열 지수'로 바꿉니다.
#    이 규칙은 데이터로 학습하지 않은 고정값입니다.
# ---------------------------------------------------------------------------
GRAY_STATS = ['img_mean', 'img_std', 'img_max', 'img_p95', 'img_p99', 'img_hot_ratio']
HEAT_STATS = ['heat_mean', 'heat_std', 'heat_p90', 'heat_p99', 'heat_max',
              'heat_warm_ratio', 'heat_hot_ratio', 'white_ratio', 'dark_ratio']
IMAGE_STATS = GRAY_STATS + HEAT_STATS


def _heat_index(rgb: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = np.asarray(rgb.convert('HSV'), dtype=np.float32) / 255.0
    hue_deg, sat, val = hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]
    # 색상 300°(자홍·보라) = 0, 0°(빨강) = 0.9 로 선형 대응. 300° 초과(분홍·자홍)는 저온으로 봅니다.
    heat = np.where(hue_deg >= 300.0, 0.0, (300.0 - hue_deg) / 300.0 * 0.9)
    white = (sat < 0.25) & (val > 0.85)          # 흰색 = 최고온
    dark = val < 0.15                            # 매우 어두운 영역 = 저온
    heat = np.where(white, 1.0, np.where(dark, 0.0, heat))
    return heat.astype(np.float32), white, dark


def _image_stats(path) -> list[float]:
    with Image.open(path) as img:
        rgb = img.convert('RGB')
    gray = np.asarray(rgb.convert('L'), dtype=np.float32) / 255.0
    heat, white, dark = _heat_index(rgb)
    return [
        float(gray.mean()), float(gray.std()), float(gray.max()),
        float(np.percentile(gray, 95)), float(np.percentile(gray, 99)),
        float((gray >= HOT_THRESHOLD).mean()),
        float(heat.mean()), float(heat.std()), float(np.percentile(heat, 90)),
        float(np.percentile(heat, 99)), float(heat.max()),
        float((heat >= 0.6).mean()), float((heat >= 0.8).mean()),
        float(white.mean()), float(dark.mean()),
    ]


def extract_image_features(frame: pd.DataFrame, paths: pd.Series, n_jobs: int = -1) -> pd.DataFrame:
    rows = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_image_stats)(p) for p in paths)
    return pd.DataFrame(rows, columns=IMAGE_STATS, index=pd.Index(frame['sample_id'].values, name='sample_id'))


# ---------------------------------------------------------------------------
# 4. 과거 방향 시계열 피처
#    같은 experiment_id 안에서 relative_time_s <= 현재 시점인 관측만 사용합니다.
#    실험 전체 통계(전체 평균·최댓값·길이·순위 등)는 사용하지 않습니다.
#    '누적 최댓값'과 '첫 관측 대비 변화량'은 실험 시작부터 현재까지의 관측만 쓰므로 미래 정보가 없습니다.
# ---------------------------------------------------------------------------
def _window_stats(t: np.ndarray, x: np.ndarray, window_s: float):
    """시간순 정렬된 한 실험에서 (t-window, t] 평균·최댓값, window 이전 값 대비 변화량·초당 기울기."""
    valid = ~np.isnan(x)
    csum = np.concatenate([[0.0], np.cumsum(np.where(valid, x, 0.0))])
    ccnt = np.concatenate([[0], np.cumsum(valid)])
    start = np.searchsorted(t, t - window_s, side='right')
    end = np.arange(1, len(t) + 1)                       # 실험·시점 중복이 없으므로 자기 자신까지
    cnt = ccnt[end] - ccnt[start]
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(cnt > 0, (csum[end] - csum[start]) / np.maximum(cnt, 1), np.nan)
    s = pd.Series(x, index=pd.to_timedelta(t, unit='s'))
    vmax = s.rolling(pd.Timedelta(seconds=window_s), min_periods=1).max().to_numpy()   # (t-w, t]
    lag = start - 1                                      # t-window 이하의 마지막 관측
    has_lag = lag >= 0
    lag_c = np.where(has_lag, lag, 0)
    delta = np.where(has_lag, x - x[lag_c], np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        slope = np.where(has_lag, delta / (t - t[lag_c]), np.nan)
    return mean, vmax, delta, slope


def _expanding_stats(t: np.ndarray, x: np.ndarray, use_min: bool):
    """직전 관측 대비 초당 기울기, 현재까지 누적 최댓값(또는 최솟값), 첫 유효 관측 대비 변화량."""
    prev = np.concatenate([[np.nan], x[:-1]])
    dt = np.concatenate([[np.nan], np.diff(t)])
    with np.errstate(invalid='ignore', divide='ignore'):
        slope_prev = (x - prev) / dt
    s = pd.Series(x)
    cum = (s.cummin() if use_min else s.cummax()).ffill().to_numpy()   # 결측 시점은 직전까지의 누적값 유지
    first_val = s.dropna().iloc[0] if s.notna().any() else np.nan       # 시간상 첫 유효 관측(현재 이전)
    from_first = x - first_val                                          # 첫 유효값 이전 행은 자동으로 NaN
    return slope_prev, cum, from_first


def add_causal_features(base: pd.DataFrame, signals: list[str], windows=WINDOWS_S) -> pd.DataFrame:
    """실험마다 따로 계산하므로 다른 실험·미래 관측이 입력에 있든 없든 결과가 같습니다."""
    blocks = {}
    times = base['relative_time_s'].to_numpy(dtype=float)
    n = len(base)
    for s in signals:
        for w in windows:
            for k in ('mean', 'max', 'delta', 'slope'):
                blocks[f'{s}_{k}{int(w)}s'] = np.full(n, np.nan)
        for k in ('slope_prev', 'cumext', 'from_first'):
            blocks[f'{s}_{k}'] = np.full(n, np.nan)
    for _, idx in base.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(times[idx], kind='mergesort')]
        t = times[idx]
        for s in signals:
            x = base[s].to_numpy(dtype=float)[idx]
            for w in windows:
                m, mx, d, sl = _window_stats(t, x, w)
                for k, v in zip(('mean', 'max', 'delta', 'slope'), (m, mx, d, sl)):
                    blocks[f'{s}_{k}{int(w)}s'][idx] = v
            sp, cum, ff = _expanding_stats(t, x, s in CUM_MIN_SIGNALS)
            blocks[f'{s}_slope_prev'][idx] = sp
            blocks[f'{s}_cumext'][idx] = cum
            blocks[f'{s}_from_first'][idx] = ff
    return pd.DataFrame(blocks, index=base.index)


# ---------------------------------------------------------------------------
# 5. 전체 피처 조립
# ---------------------------------------------------------------------------
def numeric(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for c in cols:
        out[c] = pd.to_numeric(frame[c], errors='coerce') if c in frame.columns else np.nan
    return out.replace([np.inf, -np.inf], np.nan).astype(float)


def build_features(frame: pd.DataFrame, image_features: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """frame에 들어 있는 행만으로 피처를 만듭니다. 다른 split·다른 실험의 행은 쓰지 않습니다."""
    base = frame[['sample_id', 'experiment_id']].copy()
    base['relative_time_s'] = pd.to_numeric(frame['relative_time_s'], errors='coerce')
    raw = numeric(frame, spec['numeric'])
    base[spec['numeric']] = raw
    if spec['gas']:
        gas = raw[spec['gas']]
        base['gas_zero_count'] = (gas == 0).sum(axis=1).astype(float)
        base['gas_all_zero'] = (gas == 0).all(axis=1).astype(float)
    img = image_features.reindex(frame['sample_id'].values)
    assert img.notna().all(axis=1).all(), '이미지 피처가 없는 샘플이 있습니다.'
    base[IMAGE_STATS] = img[IMAGE_STATS].values
    causal = add_causal_features(base, spec['window_signals'])
    cats = pd.DataFrame(index=frame.index)
    for c, levels in spec['categorical'].items():
        vals = frame[c].astype('string') if c in frame.columns else pd.Series(pd.NA, index=frame.index, dtype='string')
        cats[c] = pd.Categorical(vals, categories=levels)   # Train에 없던 값은 결측 처리
    gas_cols = ['gas_zero_count', 'gas_all_zero'] if spec['gas'] else []
    return pd.concat([base[['relative_time_s'] + spec['numeric'] + gas_cols + IMAGE_STATS], causal, cats], axis=1)


# ---------------------------------------------------------------------------
# 6. Test 관측 간격(5·10·15초 혼재)에 맞춘 Train 증강
#    실험마다 2번째·3번째 행만 남긴 복사본(10초·15초 간격)을 만들어 학습에 추가합니다.
#    복사본은 별도 실험으로 피처를 계산하고, 교차검증 그룹은 원래 실험 ID를 씁니다.
# ---------------------------------------------------------------------------
def subsample_experiments(frame: pd.DataFrame, step: int) -> pd.DataFrame:
    t = pd.to_numeric(frame['relative_time_s'], errors='coerce')
    order = frame.assign(_t=t).sort_values(['experiment_id', '_t'], kind='mergesort')
    rank = order.groupby('experiment_id', sort=False).cumcount()
    out = order[rank % step == 0].drop(columns='_t').copy()
    out['experiment_id'] = out['experiment_id'] + f'__every{step}'
    return out


# ---------------------------------------------------------------------------
# 7. 모델
# ---------------------------------------------------------------------------
LGBM_PARAMS = dict(
    objective='multiclass',
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,       # 같은 실험의 비슷한 행에 과적합하지 않도록 크게
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    random_state=SEED,
    deterministic=True,         # 재현성 검증용
    force_col_wise=True,
    n_jobs=4,
    verbose=-1,
)


def build_model(class_weight=None, **overrides) -> LGBMClassifier:
    return LGBMClassifier(class_weight=class_weight, **{**LGBM_PARAMS, **overrides})
