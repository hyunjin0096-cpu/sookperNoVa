"""Uni-D 배터리 열폭주 6단계 분류 — v5: v4(U-Net) + 물리 기반(Physics-informed) 지표.

v5 추가 사항 (v4 대비) — physics=True 일 때
- 열적: 온도 이동표준편차, 상승 가속도(d2T/dt2), 기울기 부호 전환, 과거 최고값 대비 차이(peak gap),
  최고값 도달 후 경과 시간, 최고값 갱신 여부, 문헌 온도 구간(70/120/180℃) 누적 체류 시간·최초 진입 후 경과 시간
- 전기·압력: 전압·압력의 이동표준편차, 가속도, 과거 최고값 대비 하락폭(전압 강하), 최고값 이후 경과 시간
- 가스: 센서별 활성 여부, 활성 센서 수, 상관 그룹(NOx·HCN·HF 계열 / CH4·CO) log 합과 그 시간 변화, 그룹 동시 활성
- 모두 같은 실험의 현재 시점 이하 관측만 사용하고, 온도 구간 값은 라벨 규칙이 아닌 입력 피처로만 씁니다.

v4 추가 사항 (v3 대비)

v4 추가 사항 (v3 대비)
- unet.py로 뽑은 U-Net 모양 피처(UNET_STATS)를 이미지 피처에 붙여 쓸 수 있습니다(--unet).
  torch가 필요한 코드는 unet.py에만 있고, 이 모듈은 torch 없이도 동작합니다.

v3 내용

v3 추가 사항 (v2 대비)
- 센서 간 상호작용 피처: 온도차(t1-t2, t1-주변, 열화상-t1), 가스 총량(log 합)과 그 시간 변화,
  온도 상승 속도×전압 변화 속도, 가스 변화×압력 변화 등. --no_interactions 로 끄고 비교 가능.
- 모델 선택: LightGBM(기본) 또는 CatBoost (--algo catboost).
- 단조성 보정: 같은 실험에서 과거→현재 예측 확률만 쓰는 전진 필터(HMM forward filtering).

v2 내용

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

# 센서 간 상호작용용 파생 신호: 이름 → (a, b) 이면 a - b
DERIVED_DIFFS = {
    't1_minus_t2': ('meas_t1', 'meas_t2'),          # 표면 - 양극부 온도
    't1_minus_atmp': ('meas_t1', 'atmp_t'),         # 표면 - 주변 온도 (발열량)
    't2_minus_atmp': ('meas_t2', 'atmp_t'),
    'thermal_minus_t1': ('thermal_t', 'meas_t1'),   # 열화상 온도 - 표면 센서
}
# 물리 기반 지표(v5)
PHYS_SIGNALS = ['meas_t1', 'meas_t2', 'meas_p', 'meas_v']   # 가속도·변동성·최고값 이후 패턴
TEMP_ZONES = (70.0, 120.0, 180.0)    # 문헌상 SEI 분해 / 분리막 손상 / 전극·전해질 분해 시작 온도(℃, 참고용 고정값)
GAS_GROUPS = {                       # EDA 상관관계 기반 그룹
    'gas_grp_nox': ['meas_hcn', 'meas_hf', 'meas_n2o', 'meas_no', 'meas_no2'],
    'gas_grp_ch4co': ['meas_ch4', 'meas_co'],
}
INTERACTION_WINDOWS = (30, 60)       # 상호작용 피처에 쓰는 기울기 윈도우(초)
# 방향 조합 플래그의 고정 임계값 (데이터로 학습하지 않음)
T_RISE = 0.01                        # ℃/s  (분당 0.6℃ 이상 상승)
V_FALL = -0.0005                     # V/s  (분당 0.03V 이상 하락)


# ---------------------------------------------------------------------------
# 1. 입력 컬럼 규격 (Train에서 한 번 정하고 모델과 함께 저장 → Test에 그대로 적용)
# ---------------------------------------------------------------------------
def infer_feature_spec(train_frame: pd.DataFrame, exclude: list[str] | None = None,
                       interactions: bool = True, unet: bool = False, physics: bool = False) -> dict:
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
    derived = []
    if interactions or physics:
        for name, (a, b) in DERIVED_DIFFS.items():
            if a in numeric_cols and b in numeric_cols:
                derived.append(name)
        if gas:
            derived.append('gas_log_sum')
    phys_gas = []
    if physics and gas:
        phys_gas = ['gas_active_count'] + [g for g, cols in GAS_GROUPS.items() if any(c in gas for c in cols)]
    window_signals = (sensors + IMAGE_WINDOW_SIGNALS + (['gas_zero_count'] if gas else []) + derived
                      + phys_gas + (UNET_WINDOW_SIGNALS if unet else []))
    image_cols = IMAGE_STATS + (UNET_STATS if unet else [])
    return {'numeric': numeric_cols, 'categorical': categorical, 'gas': gas, 'derived': derived,
            'interactions': interactions, 'unet': unet, 'image_cols': image_cols,
            'physics': physics, 'phys_gas': phys_gas,
            'window_signals': window_signals, 'ignored': ignored}


def describe_spec(spec: dict) -> str:
    return '\n'.join([
        f"  숫자 컬럼 {len(spec['numeric'])}개: {spec['numeric']}",
        f"  범주 컬럼 {len(spec['categorical'])}개: {list(spec['categorical'])}",
        f"  가스 센서 {len(spec['gas'])}개: {spec['gas']}",
        f"  파생 신호(상호작용): {spec.get('derived', [])} | 상호작용 피처 사용: {spec.get('interactions', False)}",
        f"  U-Net 모양 피처 사용: {spec.get('unet', False)} | 물리 기반 지표 사용: {spec.get('physics', False)}",
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
# U-Net(임시 마스크) 모양 피처 — 계산은 unet.py
UNET_STATS = [
    'unet_warm_area', 'unet_hot_area', 'unet_hot_in_warm',
    'unet_warm_heat_mean', 'unet_hot_heat_mean', 'unet_warm_heat_p95',
    'unet_hot_ncomp', 'unet_hot_largest_frac', 'unet_hot_spread', 'unet_hot_offset',
    'unet_warm_edge_grad', 'unet_rule_disagree',
]
UNET_WINDOW_SIGNALS = ['unet_hot_area', 'unet_hot_in_warm', 'unet_hot_spread']


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
    derived = spec.get('derived', [])
    phys_gas = spec.get('phys_gas', [])
    if phys_gas:                                          # 같은 행의 가스 값만 쓰는 물리 신호
        gas = raw[spec['gas']]
        base['gas_active_count'] = (gas > 0).sum(axis=1).astype(float).where(gas.notna().any(axis=1))
        for gname, cols in GAS_GROUPS.items():
            cols = [c for c in cols if c in spec['gas']]
            if gname in phys_gas:
                base[gname] = np.log1p(raw[cols].clip(lower=0)).sum(axis=1, min_count=1)
    for name in derived:                                 # 같은 행의 값만 쓰는 파생 신호
        if name == 'gas_log_sum':
            g = np.log1p(raw[spec['gas']].clip(lower=0))   # 가스마다 단위·규모가 달라 log로 눌러서 합산
            base[name] = g.sum(axis=1, min_count=1)
        else:
            a, b = DERIVED_DIFFS[name]
            base[name] = raw[a] - raw[b]
    image_cols = spec.get('image_cols', IMAGE_STATS)
    img = image_features.reindex(frame['sample_id'].values)
    assert img[IMAGE_STATS].notna().all(axis=1).all(), '이미지 피처가 없는 샘플이 있습니다.'
    assert set(image_cols) <= set(img.columns), 'U-Net 피처가 없습니다. --unet 가중치를 확인하십시오.'
    base[image_cols] = img[image_cols].values            # U-Net 피처의 NaN(고온 영역 없음 등)은 그대로 둠
    causal = add_causal_features(base, spec['window_signals'])
    cats = pd.DataFrame(index=frame.index)
    for c, levels in spec['categorical'].items():
        vals = frame[c].astype('string') if c in frame.columns else pd.Series(pd.NA, index=frame.index, dtype='string')
        cats[c] = pd.Categorical(vals, categories=levels)   # Train에 없던 값은 결측 처리
    gas_cols = ['gas_zero_count', 'gas_all_zero'] if spec['gas'] else []
    parts = [base[['relative_time_s'] + spec['numeric'] + gas_cols + derived + phys_gas + image_cols], causal]
    if spec.get('interactions', False):
        parts.append(interaction_features(causal))
    if spec.get('physics', False):
        parts.append(physics_features(base, raw, causal, spec))
    parts.append(cats)
    return pd.concat(parts, axis=1)


def _phys_one(t: np.ndarray, x: np.ndarray, zones: tuple) -> dict:
    """한 실험(시간순)의 한 신호에 대한 물리 지표. 모두 현재 시점 이하 관측만 사용."""
    n = len(t)
    valid = ~np.isnan(x)
    out = {}
    # 이동표준편차 (t-w, t]
    xv = np.where(valid, x, 0.0)
    c1 = np.concatenate([[0.0], np.cumsum(xv)])
    c2 = np.concatenate([[0.0], np.cumsum(xv * xv)])
    cn = np.concatenate([[0], np.cumsum(valid)])
    end = np.arange(1, n + 1)
    for w in (30.0, 60.0):
        st = np.searchsorted(t, t - w, side='right')
        k = cn[end] - cn[st]
        with np.errstate(invalid='ignore', divide='ignore'):
            m = (c1[end] - c1[st]) / k
            var = (c2[end] - c2[st]) / k - m * m
        out[f'std{int(w)}s'] = np.where(k >= 2, np.sqrt(np.clip(var, 0, None)), np.nan)
    # 가속도: 최근 30초 기울기와 그 이전 30초 기울기의 차이 / 시간
    l1 = np.searchsorted(t, t - 30.0, side='right') - 1
    l2 = np.searchsorted(t, t - 60.0, side='right') - 1
    ok = (l1 >= 0) & (l2 >= 0)
    a, b = np.where(ok, l1, 0), np.where(ok, l2, 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        s_now = (x - x[a]) / (t - t[a])
        s_prev = (x[a] - x[b]) / (t[a] - t[b])
        acc = (s_now - s_prev) / ((t - t[b]) / 2.0)
    ok &= np.isfinite(acc)
    out['accel'] = np.where(ok, acc, np.nan)
    out['slope_flip'] = np.where(ok, (np.sign(s_now) * np.sign(s_prev) < 0).astype(float), np.nan)
    # 과거 최고값 대비 차이, 최고값 이후 경과 시간, 최고값 갱신 여부
    pastmax = pd.Series(x).cummax().ffill().to_numpy()
    is_peak = valid & (x >= pastmax - 1e-12)
    peak_t = pd.Series(np.where(is_peak, t, np.nan)).ffill().to_numpy()
    out['peak_gap'] = pastmax - x
    out['since_peak'] = t - peak_t
    out['new_peak'] = np.where(valid, is_peak.astype(float), np.nan)
    # 문헌 온도 구간: 누적 체류 시간, 최초 진입 후 경과 시간
    dt = np.concatenate([[0.0], np.diff(t)])
    for z in zones:
        above = valid & (x >= z)
        out[f'dwell{int(z)}'] = np.cumsum(np.where(above, dt, 0.0))
        first_t = pd.Series(np.where(above, t, np.nan)).cummin().ffill().to_numpy()   # 지금까지 처음 넘은 시각
        out[f'since{int(z)}'] = t - first_t
    return out


def physics_features(base: pd.DataFrame, raw: pd.DataFrame, causal: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """물리 기반 지표(ph_ 접두사). 실험마다 시간순으로 따로 계산합니다."""
    times = base['relative_time_s'].to_numpy(dtype=float)
    n = len(base)
    blocks = {}
    signals = [s for s in PHYS_SIGNALS if s in base.columns]
    for _, idx in base.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(times[idx], kind='mergesort')]
        t = times[idx]
        for s in signals:
            zones = TEMP_ZONES if s in ('meas_t1', 'meas_t2') else ()
            for k, v in _phys_one(t, base[s].to_numpy(dtype=float)[idx], zones).items():
                name = f'ph_{s}_{k}'
                if name not in blocks:
                    blocks[name] = np.full(n, np.nan)
                blocks[name][idx] = v
    out = pd.DataFrame(blocks, index=base.index)
    if 'meas_v' in base.columns:                          # 전압 강하율(과거 최고 전압 대비 비율)
        vmax = out['ph_meas_v_peak_gap'] + base['meas_v']
        out['ph_meas_v_drop_ratio'] = out['ph_meas_v_peak_gap'] / vmax.where(vmax.abs() > 1e-6)
    # 동시 발생 플래그 (과거 30초 기울기, 고정 임계값. 압력은 단위가 불명확해 방향만 사용)
    def flag(cond, *cols):
        return cond.astype(float).where(pd.concat(cols, axis=1).notna().all(axis=1))
    sl = lambda sig: causal.get(f'{sig}_slope30s')
    t1, v, p = sl('meas_t1'), sl('meas_v'), sl('meas_p')
    if t1 is not None and v is not None:
        out['ph_t1up_vdown30s'] = flag((t1 > T_RISE) & (v < V_FALL), t1, v)
    if t1 is not None and p is not None:
        out['ph_t1up_pup30s'] = flag((t1 > T_RISE) & (p > 0), t1, p)
    gas = spec['gas']
    if gas:                                               # 센서별 활성 여부, 그룹 동시 활성
        g = raw[gas]
        for c in gas:
            out[f'ph_{c}_active'] = (g[c] > 0).astype(float).where(g[c].notna())
        for gname, cols in GAS_GROUPS.items():
            cols = [c for c in cols if c in gas]
            if cols:
                out[f'ph_{gname}_all_active'] = (g[cols] > 0).all(axis=1).astype(float).where(g[cols].notna().all(axis=1))
    return out


def interaction_features(causal: pd.DataFrame) -> pd.DataFrame:
    """여러 센서가 '동시에 어떤 방향으로' 변하는지를 나타내는 피처.
    모두 과거 윈도우 기울기(인과 피처)의 조합이라 현재 시점 이하 관측만 사용합니다."""
    out = {}
    def col(name):
        return causal[name] if name in causal.columns else None
    def both_flag(cond, a, b):
        return cond.astype(float).where(a.notna() & b.notna())
    for w in INTERACTION_WINDOWS:
        t1, t2 = col(f'meas_t1_slope{w}s'), col(f'meas_t2_slope{w}s')
        v, p = col(f'meas_v_slope{w}s'), col(f'meas_p_slope{w}s')
        g = col(f'gas_log_sum_slope{w}s')
        if t1 is not None and v is not None:
            out[f'ix_t1v_prod{w}s'] = t1 * v                           # 온도↑ × 전압↓ 이면 음수로 커짐
            out[f'ix_t1_per_v{w}s'] = t1 / (v.abs() + 1e-3)
            out[f'ix_t1up_vdown{w}s'] = both_flag((t1 > T_RISE) & (v < V_FALL), t1, v)
        if t1 is not None and t2 is not None:
            out[f'ix_t1_t2_slope_diff{w}s'] = t1 - t2                  # 어느 쪽이 더 빨리 데워지는가
        if t1 is not None and p is not None:
            out[f'ix_t1p_prod{w}s'] = t1 * p
        if g is not None and p is not None:
            out[f'ix_gasp_prod{w}s'] = g * p                           # 가스↑ × 압력↑ (벤팅 계열 신호)
            out[f'ix_gasup_pup{w}s'] = both_flag((g > 0) & (p > 0), g, p)
        if g is not None and t1 is not None:
            out[f'ix_gas_t1_prod{w}s'] = g * t1
    # 가속도: 최근 30초 기울기 - 최근 120초 기울기 (상승이 빨라지는 중인가)
    for s in ('meas_t1', 'meas_t2', 'meas_p', 'gas_log_sum', 'meas_v'):
        a, b = col(f'{s}_slope30s'), col(f'{s}_slope120s')
        if a is not None and b is not None:
            out[f'ix_{s}_accel'] = a - b
    return pd.DataFrame(out, index=causal.index)


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


CATBOOST_PARAMS = dict(
    loss_function='MultiClass',
    iterations=800,
    learning_rate=0.08,
    border_count=128,
    depth=6,
    l2_leaf_reg=5.0,
    random_seed=SEED,
    thread_count=8,
    allow_writing_files=False,
    verbose=0,
)


class CatBoostModel:
    """CatBoost를 LightGBM과 같은 방식(fit/predict/predict_proba)으로 쓰기 위한 래퍼."""

    def __init__(self, class_weight=None, **params):
        from catboost import CatBoostClassifier
        if class_weight == 'balanced':
            params['auto_class_weights'] = 'Balanced'
        self.model = CatBoostClassifier(**params)

    @staticmethod
    def _prep(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in X.columns:
            if isinstance(X[c].dtype, pd.CategoricalDtype):
                X[c] = X[c].cat.codes.replace(-1, np.nan).astype(float)
        return X

    def fit(self, X, y):
        self.model.fit(self._prep(X), y)
        self.classes_ = np.asarray(self.model.classes_).astype(int)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._prep(X))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    @property
    def feature_importances_(self):
        return self.model.get_feature_importance()


def build_model(class_weight=None, algo: str = 'lgbm', **overrides):
    if algo == 'catboost':
        return CatBoostModel(class_weight, **{**CATBOOST_PARAMS, **overrides})
    return LGBMClassifier(class_weight=class_weight, **{**LGBM_PARAMS, **overrides})


# ---------------------------------------------------------------------------
# 8. 단조성 보정 — 전진 필터(HMM forward filtering)
#    Train에서 시간순 라벨 감소가 0건이므로 '단계는 내려가지 않는다'는 사전 지식을 반영합니다.
#    같은 실험 안에서 시간순으로, 이전 시점까지의 믿음(belief)에 전이행렬을 곱하고
#    현재 시점의 모델 확률을 곱해 갱신합니다. 미래 행은 전혀 쓰지 않으므로
#    미래 행 제거·실험 분리·행 순서 변경에도 결과가 같습니다.
#    - 전이행렬: Train 라벨의 5초 간격 전이 빈도로 추정. 간격이 dt초면 (dt/5)제곱.
#      2단계 이상 점프(예: 4→6)도 Train 빈도대로 작은 확률로 허용됩니다.
#    - down_eps: 아래 단계로 돌아갈 아주 작은 확률. 초반 오탐 하나가 끝까지 고정되는 것을 막습니다.
#    - alpha: 모델 확률의 반영 강도(작을수록 과거 믿음을 더 신뢰).
# ---------------------------------------------------------------------------
BASE_DT = 5.0


def estimate_transitions(frame: pd.DataFrame, pseudo: float = 1.0) -> dict:
    """Train 라벨로 5초 간격 전이행렬과 첫 관측 분포를 추정합니다(라벨은 학습 데이터에서만 사용)."""
    k = len(LABELS)
    counts = np.zeros((k, k))
    first = np.zeros(k)
    t = pd.to_numeric(frame['relative_time_s'], errors='coerce')
    lab = pd.to_numeric(frame['label']).astype(int).to_numpy() - 1
    order = frame.assign(_t=t, _y=lab).sort_values(['experiment_id', '_t'], kind='mergesort')
    for _, g in order.groupby('experiment_id', sort=False):
        y = g['_y'].to_numpy()
        first[y[0]] += 1
        np.add.at(counts, (y[:-1], y[1:]), 1)
    upper = np.triu(np.ones((k, k)))                    # 같은 단계 유지 + 상승만 의사빈도 부여
    counts = counts + pseudo * upper
    trans = counts / counts.sum(axis=1, keepdims=True)
    pi = (first + pseudo) / (first + pseudo).sum()
    return {'trans': trans, 'pi': pi}


def _step_matrix(trans: np.ndarray, down_eps: float, steps: int, cache: dict) -> np.ndarray:
    key = steps
    if key not in cache:
        m = trans + down_eps * np.tril(np.ones_like(trans), -1)
        m = m / m.sum(axis=1, keepdims=True)
        cache[key] = np.linalg.matrix_power(m, steps)
    return cache[key]


def forward_filter(proba: np.ndarray, experiment_id, times, params: dict) -> np.ndarray:
    """proba: (n, 6) 모델 확률(행 순서 그대로). 반환: 보정된 확률(같은 행 순서)."""
    proba = np.asarray(proba, dtype=float)
    out = np.empty_like(proba)
    times = np.asarray(times, dtype=float)
    trans, pi = np.asarray(params['trans']), np.asarray(params['pi'])
    alpha, down_eps = params['alpha'], params['down_eps']
    cache = {}
    emit = np.clip(proba, 1e-9, 1.0) ** alpha
    groups = pd.Series(np.arange(len(proba))).groupby(np.asarray(experiment_id), sort=False).indices
    for _, idx in groups.items():
        idx = idx[np.argsort(times[idx], kind='mergesort')]
        belief = None
        prev_t = None
        for i in idx:
            if belief is None:
                b = pi * emit[i]
            else:
                steps = max(1, int(round((times[i] - prev_t) / BASE_DT)))
                b = (belief @ _step_matrix(trans, down_eps, steps, cache)) * emit[i]
            belief = b / b.sum()
            out[i] = belief
            prev_t = times[i]
    return out


def tune_postprocess(proba, labels, experiment_id, times, trans_info: dict,
                     alphas=(0.3, 0.5, 0.7, 1.0, 1.5), down_eps_list=(0.0, 1e-4, 1e-3, 1e-2)):
    """Train OOF 확률로 alpha·down_eps를 고릅니다(Validation은 쓰지 않음)."""
    from sklearn.metrics import f1_score
    labels = np.asarray(labels)
    best, table = None, []
    for a in alphas:
        for e in down_eps_list:
            params = {**trans_info, 'alpha': a, 'down_eps': e}
            pred = np.array(LABELS)[forward_filter(proba, experiment_id, times, params).argmax(axis=1)]
            f1 = f1_score(labels, pred, labels=LABELS, average='macro', zero_division=0)
            table.append((a, e, f1))
            if best is None or f1 > best[2]:
                best = (a, e, f1)
    return {**trans_info, 'alpha': best[0], 'down_eps': best[1]}, best[2], table
