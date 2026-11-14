"""Uni-D 배터리 열폭주 6단계 분류 — 개선 모델 공통 모듈 (v8).

베이스라인 v7 대비 변경점
  1. 로지스틱 회귀 → LightGBM
  2. 이미지: 흑백 통계만 → 의사색(RGB) 색상 구간 비율 추가 (흑백 변환은 색 순서를 왜곡함)
  3. 센서: meas_v/t1/t2/p + 가스 센서(meas_*) + thermal_t + 파생값(t_diff, 가스 합/0 비율)
  4. 시계열: 30/90/180초 윈도우 (평균·최댓값·최솟값·변화율) + 현재까지 누적 최대/최소
  5. Test 관측 간격(5·10·15초) 대응: 학습 데이터를 2·3칸씩 솎아낸 복제본도 함께 학습
  6. (선택) 과거 방향 전진 필터: 단계는 되돌아가지 않는다는 성질 사용

규정 준수 원칙: 모든 피처는 같은 experiment_id 안에서 relative_time_s <= 현재 시점인 관측만 사용합니다.
실험마다 따로 계산하므로 다른 실험·미래 행이 입력에 있든 없든 같은 샘플은 같은 결과가 나옵니다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

# ---------------------------------------------------------------------------
# 설정 (운영진 확인이 필요한 항목은 아래 플래그로 끌 수 있습니다)
# ---------------------------------------------------------------------------
SEED = 42
LABELS = list(range(1, 7))

CORE_SENSORS = ['meas_v', 'meas_t1', 'meas_t2', 'meas_p']      # meas_a는 전체 split에서 상수 → 제외
EXCLUDE_MEAS = {'meas_a', 'meas_t1_plus_t2'}                    # 상수 / t1+t2 단순 합(중복)
USE_GAS = True            # meas_ 로 시작하는 나머지 센서(가스 등) 사용 여부
USE_THERMAL_T = True      # thermal_t 컬럼 사용 여부
USE_META = False          # bat_class, bat_cc, soc_status, tr_method (실험 조건 컬럼) — 운영진 확인 전까지 끔
META_COLS = ['bat_class', 'bat_cc', 'soc_status', 'tr_method']
USE_CUMULATIVE = True     # 현재까지의 누적 최대/최소 (과거+현재 관측만 사용)
USE_TIME = False          # relative_time_s 원값을 입력으로 쓸지 (실험마다 진행 속도가 달라 기본은 끔)

WINDOWS_S = (30.0, 90.0, 180.0)   # 과거 윈도우 길이(초). 행 수가 아니라 relative_time_s 기준
AUG_STRIDES = (2, 3)              # 학습 데이터를 2칸·3칸씩 솎은 복제본 추가 (Test에 10초·15초 간격이 있음)
BASE_STEP_S = 5.0                 # Train/Validation 관측 간격

HOT_THRESHOLD = 0.8
IMG_SIZE = 96
IMG_VERSION = 'v8'                # 이미지 피처 정의가 바뀌면 올려서 캐시를 무효화합니다.

GRAY_STATS = ['img_mean', 'img_std', 'img_max', 'img_p95', 'img_p99', 'img_hot_ratio']
HUE_BINS = [f'img_hue{i}' for i in range(12)]
IMAGE_STATS = GRAY_STATS + ['img_dark', 'img_white', 'img_gray'] + HUE_BINS + ['img_warm']
IMAGE_WINDOW_SIGNALS = ['img_mean', 'img_max', 'img_p99', 'img_hot_ratio', 'img_white', 'img_warm']


# ---------------------------------------------------------------------------
# 1. 이미지 경로 연결
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
# 2. 이미지 피처 (샘플마다 독립 계산 → 다른 행 정보 사용 없음)
#    열화상은 온도-색상 변환표가 없는 의사색 RGB 이므로, 흑백 밝기 통계 + 색상 구간별 픽셀 비율을 함께 씁니다.
#    색상 순서(어느 색이 더 뜨거운지)는 모델이 학습하도록 두고, 사람이 정한 순서를 강제하지 않습니다.
# ---------------------------------------------------------------------------
def _image_stats(path) -> list[float]:
    with Image.open(path) as img:
        rgb = img.convert('RGB')
        gray = np.asarray(rgb.convert('L'), dtype=np.float32) / 255.0
        small = rgb.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        hsv = np.asarray(small.convert('HSV'), dtype=np.float32)
    out = [
        float(gray.mean()), float(gray.std()), float(gray.max()),
        float(np.percentile(gray, 95)), float(np.percentile(gray, 99)),
        float((gray >= HOT_THRESHOLD).mean()),
    ]
    h = hsv[..., 0] * (360.0 / 255.0)
    s, v = hsv[..., 1], hsv[..., 2]
    dark = v < 50
    white = (~dark) & (s < 50) & (v >= 200)
    grayish = (~dark) & (s < 50) & (~white)
    colored = (~dark) & (s >= 50)
    n = float(h.size)
    hue_bin = (((h + 15.0) % 360.0) // 30.0).astype(int)        # 0=빨강 1=주황 2=노랑 ... 12구간
    hue_frac = [float((colored & (hue_bin == b)).sum() / n) for b in range(12)]
    warm = float((white | (colored & (hue_bin <= 2))).sum() / n)  # 흰색 + 빨강·주황·노랑
    out += [float(dark.mean()), float(white.mean()), float(grayish.mean())] + hue_frac + [warm]
    return out


def extract_image_features(frame: pd.DataFrame, paths: pd.Series, n_jobs: int = -1) -> pd.DataFrame:
    rows = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_image_stats)(p) for p in paths)
    return pd.DataFrame(rows, columns=IMAGE_STATS, index=pd.Index(frame['sample_id'].values, name='sample_id'))


# ---------------------------------------------------------------------------
# 3. 과거 방향 시계열 피처
# ---------------------------------------------------------------------------
def _causal_one_experiment(t: np.ndarray, X: np.ndarray, windows, cumulative: bool) -> np.ndarray:
    """t: 시간순 정렬된 초 단위 시각, X: (n, m) 신호. 행마다 (t-w, t] 구간 통계와 누적 통계를 계산합니다."""
    df = pd.DataFrame(X, index=pd.to_timedelta(t, unit='s'))
    blocks = []
    for w in windows:
        r = df.rolling(f'{int(w)}s', closed='right')             # (t-w, t] : 현재 포함, 미래 제외
        blocks += [r.mean().to_numpy(), r.max().to_numpy(), r.min().to_numpy()]
        lag = np.searchsorted(t, t - w, side='right') - 1         # t-w 이하의 마지막 관측
        ok = lag >= 0
        safe = np.clip(lag, 0, None)
        dt = np.where(ok, t - t[safe], np.nan)
        with np.errstate(invalid='ignore', divide='ignore'):
            rate = np.where(ok[:, None], (X - X[safe]) / dt[:, None], np.nan)   # 초당 변화율
        blocks.append(rate)
    if cumulative:
        blocks += [df.cummax().to_numpy(), df.cummin().to_numpy()]
    return np.concatenate(blocks, axis=1)


def causal_feature_names(signals, windows, cumulative: bool) -> list[str]:
    names = []
    for w in windows:
        for agg in ('mean', 'max', 'min', 'rate'):
            names += [f'{s}_{agg}{int(w)}s' for s in signals]
    if cumulative:
        for agg in ('cummax', 'cummin'):
            names += [f'{s}_{agg}' for s in signals]
    return names


def add_causal_features(base: pd.DataFrame, signals: list[str], windows=WINDOWS_S, cumulative=USE_CUMULATIVE) -> pd.DataFrame:
    names = causal_feature_names(signals, windows, cumulative)
    out = np.full((len(base), len(names)), np.nan)
    times = base['relative_time_s'].to_numpy(dtype=float)
    values = base[signals].to_numpy(dtype=float)
    for _, idx in base.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(times[idx], kind='mergesort')]
        out[idx] = _causal_one_experiment(times[idx], values[idx], windows, cumulative)
    return pd.DataFrame(out, index=base.index, columns=names)


# ---------------------------------------------------------------------------
# 4. 전체 피처 조립
# ---------------------------------------------------------------------------
def numeric(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return frame[cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)


def gas_columns(frame: pd.DataFrame) -> list[str]:
    """meas_ 로 시작하는 나머지 센서(가스 등). 어떤 컬럼이 잡혔는지 train.py 가 출력합니다."""
    if not USE_GAS:
        return []
    return sorted(c for c in frame.columns
                  if c.startswith('meas_') and c not in CORE_SENSORS and c not in EXCLUDE_MEAS)


def build_features(frame: pd.DataFrame, image_features: pd.DataFrame) -> pd.DataFrame:
    """frame에 들어 있는 행만으로 피처를 만듭니다. 다른 split·다른 실험의 행은 쓰지 않습니다."""
    gas = gas_columns(frame)
    sensors = CORE_SENSORS + (['thermal_t'] if USE_THERMAL_T and 'thermal_t' in frame.columns else [])

    base = frame[['sample_id', 'experiment_id']].copy()
    base['relative_time_s'] = pd.to_numeric(frame['relative_time_s'], errors='coerce')
    base[sensors] = numeric(frame, sensors)
    base['t_diff'] = base['meas_t1'] - base['meas_t2']
    derived = ['t_diff']
    if gas:
        g = numeric(frame, gas)
        base[gas] = g
        base['gas_zero_frac'] = (g == 0).sum(axis=1) / len(gas)   # 가스 센서가 동시에 0인 정도 (EDA: 6단계에서 많음)
        base['gas_sum'] = g.sum(axis=1, min_count=1)
        derived += ['gas_zero_frac', 'gas_sum']

    img = image_features.reindex(frame['sample_id'].values)
    assert img.notna().all(axis=1).all(), '이미지 피처가 없는 샘플이 있습니다.'
    base[IMAGE_STATS] = img[IMAGE_STATS].values

    signals = sensors + derived + IMAGE_WINDOW_SIGNALS
    causal = add_causal_features(base, signals)

    raw = sensors + derived + gas + IMAGE_STATS
    if USE_TIME:
        raw = ['relative_time_s'] + raw
    parts = [base[raw], causal]
    if USE_META:
        parts.append(numeric(frame, META_COLS))
    return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# 5. 관측 간격 대응용 데이터 솎아내기 (학습 전용)
# ---------------------------------------------------------------------------
def subsample_stride(frame: pd.DataFrame, stride: int) -> pd.DataFrame:
    """실험마다 시간순으로 stride 칸에 한 행씩 남깁니다. (Test 의 10초·15초 간격 흉내)"""
    f = frame.sort_values(['experiment_id', 'relative_time_s'], kind='mergesort')
    keep = (f.groupby('experiment_id').cumcount() % stride) == 0
    return f[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. 모델
# ---------------------------------------------------------------------------
def build_model(class_weight=None, n_estimators: int = 300):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        objective='multiclass', n_estimators=n_estimators, learning_rate=0.05,
        num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.5, reg_lambda=1.0, class_weight=class_weight,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )


# ---------------------------------------------------------------------------
# 7. 과거 방향 전진 필터 (선택)
#    단계는 실험 안에서 거의 되돌아가지 않으므로, 이전 시점의 신념을 전이행렬로 전파한 뒤
#    현재 분류기 확률을 곱합니다. 미래를 쓰지 않는 forward 방식이며 실험마다 새로 시작합니다.
# ---------------------------------------------------------------------------
def estimate_filter_stats(frame: pd.DataFrame, labels) -> dict:
    f = frame[['experiment_id', 'relative_time_s']].copy()
    f['label'] = np.asarray(labels, dtype=int)
    f = f.sort_values(['experiment_id', 'relative_time_s'], kind='mergesort')
    counts = np.full((6, 6), 1e-3)
    first = np.ones(6)
    for _, g in f.groupby('experiment_id', sort=False):
        lab = g['label'].to_numpy()
        first[lab[0] - 1] += 1
        for a, b in zip(lab[:-1], lab[1:]):
            counts[a - 1, b - 1] += 1
    prior = np.bincount(f['label'].to_numpy() - 1, minlength=6).astype(float) + 1.0
    return {'T': counts / counts.sum(1, keepdims=True), 'pi0': first / first.sum(), 'prior': prior / prior.sum()}


def causal_filter(proba: np.ndarray, frame: pd.DataFrame, stats: dict, tau: float) -> np.ndarray:
    t = pd.to_numeric(frame['relative_time_s']).to_numpy(dtype=float)
    out = np.empty(len(frame), dtype=int)
    cache: dict[int, np.ndarray] = {}
    for _, idx in frame.reset_index(drop=True).groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        belief, prev = None, None
        for i in idx:
            lik = (np.clip(proba[i], 1e-6, None) / stats['prior']) ** tau
            if belief is None:
                belief = stats['pi0'] * lik
            else:
                k = max(1, int(round((t[i] - prev) / BASE_STEP_S)))
                if k not in cache:
                    cache[k] = np.linalg.matrix_power(stats['T'], k)
                belief = (belief @ cache[k]) * lik
            belief = belief / belief.sum()
            out[i] = int(np.argmax(belief)) + 1
            prev = t[i]
    return out


def predict_labels(bundle: dict, frame: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    model = bundle['pipeline']
    proba = model.predict_proba(X[bundle['features']])
    filt = bundle.get('filter')
    if filt is None:
        return np.asarray(model.classes_)[proba.argmax(1)].astype(int)
    return causal_filter(proba, frame, filt['stats'], filt['tau'])
