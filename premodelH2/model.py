"""Uni-D 배터리 열폭주 6단계 분류 — 공통 모듈 v10.

설계 근거 (문헌 + 본 대회 EDA)
  A. 순서형 분해 (Frank & Hall): 6-클래스를 "k단계 초과인가?" 이진 문제 5개로 나눠 학습.
     단계는 순서가 있으므로, 경계 하나씩 전담시키면 인접 단계(2↔3, 4↔5) 혼동이 줄어듭니다.
     이 확률을 일반 다중분류 확률과 섞습니다(가중치는 OOF로 선택).
  B. 실험 상대 정규화: 배터리 형태·용량·유도 방식이 실험마다 달라 절대 온도 기준이 통하지 않습니다
     (EDA: 5단계 thermal_t 중앙값이 가열 40.6℃ vs 과충전 308.9℃).
     첫 관측 대비 변화, 현재까지 최댓값 대비 비율/하락폭, 최고값 이후 경과 시간을 씁니다.
     모두 같은 실험의 과거·현재 관측만 사용합니다.
  C. 핫스팟 동역학 이미지 피처: 밝기 통계보다 고온 영역의 면적·중심·퍼짐·엔트로피·경계 세기가
     단계 구분에 유효하다는 보고를 반영했습니다. 샘플마다 독립 계산합니다.
  D. 과적합 억제: 피처 상위 N개만 사용, 시드 앙상블, 규제 강화.
     CNN은 본 대회 데이터에서 성능을 떨어뜨려 기본 비활성화입니다(USE_CNN).

규정 준수: 모든 피처는 같은 experiment_id 안에서 relative_time_s <= 현재인 관측만 사용합니다.
실험마다 독립 계산하므로 다른 실험·미래 행이 입력에 있든 없든 같은 샘플은 같은 결과가 나옵니다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

SEED = 42
LABELS = list(range(1, 7))

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
CORE_SENSORS = ['meas_v', 'meas_t1', 'meas_t2', 'meas_p']   # meas_a는 전체 split 상수 → 제외
EXCLUDE_MEAS = {'meas_a', 'meas_t1_plus_t2'}
USE_GAS = True            # meas_* 나머지(가스 센서 10종)
USE_THERMAL_T = True
USE_META = False          # bat_class/bat_cc/soc_status/tr_method — 운영진 확인 전까지 끔
META_COLS = ['bat_class', 'bat_cc', 'soc_status', 'tr_method']
USE_TIME = False          # relative_time_s 원값 (실험마다 진행 속도가 달라 기본 끔)
USE_RELATIVE = True       # B. 실험 상대 정규화
USE_EVENTS = True         # 임계값 도달 후 경과 시간
USE_CNN = False           # CNN 확률을 피처로 쓸지 (기본 끔)

EVENTS = [('meas_v', '<', 3.0), ('meas_v', '<', 1.0), ('meas_v', '<', 0.1),
          ('meas_t1', '>=', 80.0), ('meas_t1', '>=', 150.0), ('meas_t1', '>=', 250.0)]

WINDOWS_S = (30.0, 90.0, 180.0)
AUG_STRIDES = (2, 3)
BASE_STEP_S = 5.0

IMG_VERSION = 'v10'
HOT_THRESHOLD = 0.8
HOT_QUANTILES = (0.90, 0.97, 0.99)   # 상위 q 밝기 영역을 "고온 영역"으로 보고 동역학을 잽니다

GRAY_STATS = ['img_mean', 'img_std', 'img_max', 'img_p95', 'img_p99', 'img_hot_ratio',
              'img_grad', 'img_entropy']
HOT_STATS = [f'img_hot{int(q * 100)}_{k}' for q in HOT_QUANTILES
             for k in ('area', 'cx', 'cy', 'spread', 'val')]
HUE_BINS = [f'img_hue{i}' for i in range(12)]
IMAGE_STATS = GRAY_STATS + HOT_STATS + ['img_dark', 'img_white', 'img_gray'] + HUE_BINS + ['img_warm']
IMAGE_WINDOW_SIGNALS = ['img_mean', 'img_p99', 'img_hot_ratio', 'img_white', 'img_warm',
                        'img_hot97_area', 'img_hot97_spread', 'img_grad']

CNN_COLS = [f'cnn_p{k}' for k in range(1, 7)] + ['cnn_exp']
CNN_WINDOW_SIGNALS = ['cnn_exp', 'cnn_p2', 'cnn_p4']

IMG_SIZE = 96


# ---------------------------------------------------------------------------
# 1. 이미지 경로
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
# 2. 이미지 피처 — 샘플마다 독립 계산 (다른 행 정보 없음)
#    좌표는 이미지 크기로 나눠 0~1로 쓰므로 실험마다 다른 해상도에 영향을 덜 받습니다.
# ---------------------------------------------------------------------------
def _image_stats(path) -> list[float]:
    with Image.open(path) as img:
        rgb = img.convert('RGB')
        gray_full = np.asarray(rgb.convert('L'), dtype=np.float32) / 255.0
        small = rgb.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        g = np.asarray(small.convert('L'), dtype=np.float32) / 255.0
        hsv = np.asarray(small.convert('HSV'), dtype=np.float32)

    gy, gx = np.gradient(g)
    grad = float(np.sqrt(gx ** 2 + gy ** 2).mean())
    hist = np.histogram(g, bins=32, range=(0.0, 1.0))[0].astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())

    out = [float(gray_full.mean()), float(gray_full.std()), float(gray_full.max()),
           float(np.percentile(gray_full, 95)), float(np.percentile(gray_full, 99)),
           float((gray_full >= HOT_THRESHOLD).mean()), grad, entropy]

    # 핫스팟 동역학: 상위 q 밝기 영역의 면적·무게중심·퍼짐·평균 밝기
    ys, xs = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    ys = ys / (IMG_SIZE - 1.0)
    xs = xs / (IMG_SIZE - 1.0)
    for q in HOT_QUANTILES:
        thr = float(np.quantile(g, q))
        mask = g >= thr
        n = float(mask.sum())
        if n < 1:
            out += [0.0, 0.5, 0.5, 0.0, thr]
            continue
        mx, my = float(xs[mask].mean()), float(ys[mask].mean())
        spread = float(np.sqrt(((xs[mask] - mx) ** 2 + (ys[mask] - my) ** 2).mean()))
        out += [n / g.size, mx, my, spread, float(g[mask].mean())]

    h = hsv[..., 0] * (360.0 / 255.0)
    s, v = hsv[..., 1], hsv[..., 2]
    dark = v < 50
    white = (~dark) & (s < 50) & (v >= 200)
    grayish = (~dark) & (s < 50) & (~white)
    colored = (~dark) & (s >= 50)
    size = float(h.size)
    hue_bin = (((h + 15.0) % 360.0) // 30.0).astype(int)
    out += [float(dark.mean()), float(white.mean()), float(grayish.mean())]
    out += [float((colored & (hue_bin == b)).sum() / size) for b in range(12)]
    out += [float((white | (colored & (hue_bin <= 2))).sum() / size)]
    return out


def extract_image_features(frame: pd.DataFrame, paths: pd.Series, n_jobs: int = -1) -> pd.DataFrame:
    rows = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_image_stats)(p) for p in paths)
    return pd.DataFrame(rows, columns=IMAGE_STATS,
                        index=pd.Index(frame['sample_id'].astype(str).values, name='sample_id'))


# ---------------------------------------------------------------------------
# 3. 과거 방향 시계열 피처
# ---------------------------------------------------------------------------
def _first_hit_elapsed(t: np.ndarray, x: np.ndarray, op: str, thr: float) -> np.ndarray:
    with np.errstate(invalid='ignore'):
        hit = (x < thr) if op == '<' else (x >= thr)
    hit &= ~np.isnan(x)
    first = np.minimum.accumulate(np.where(hit, t, np.inf))
    return np.where(np.isfinite(first), t - first, np.nan)


def _relative_blocks(t: np.ndarray, X: np.ndarray) -> list[np.ndarray]:
    """실험 상대 정규화: 시작 대비 변화 / 최댓값 대비 비율 / 최댓값 대비 하락 / 최고값 이후 경과 시간."""
    n, m = X.shape
    from_start = np.full((n, m), np.nan)
    ratio_max = np.full((n, m), np.nan)
    drop_max = np.full((n, m), np.nan)
    since_max = np.full((n, m), np.nan)
    span = np.full((n, m), np.nan)
    for j in range(m):
        x = X[:, j]
        valid = ~np.isnan(x)
        if not valid.any():
            continue
        fi = int(np.argmax(valid))
        from_start[fi:, j] = x[fi:] - x[fi]
        cmax = np.fmax.accumulate(np.where(valid, x, -np.inf))
        cmin = np.fmin.accumulate(np.where(valid, x, np.inf))
        cmax[:fi] = np.nan
        cmin[:fi] = np.nan
        rng = cmax - cmin
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio_max[:, j] = np.where(np.abs(cmax) > 1e-9, x / cmax, np.nan)
            span[:, j] = np.where(rng > 1e-9, (x - cmin) / rng, np.nan)
        drop_max[:, j] = cmax - x
        mark = np.where(valid & (x >= cmax), t, np.nan)
        since_max[:, j] = t - pd.Series(mark).ffill().to_numpy()
    return [from_start, ratio_max, drop_max, since_max, span]


def _causal_one_experiment(t: np.ndarray, X: np.ndarray, windows) -> np.ndarray:
    df = pd.DataFrame(X, index=pd.to_timedelta(t, unit='s'))
    blocks = []
    for w in windows:
        r = df.rolling(f'{int(w)}s', closed='right')      # (t-w, t] : 현재 포함, 미래 제외
        blocks += [r.mean().to_numpy(), r.max().to_numpy(), r.min().to_numpy(), r.std().to_numpy()]
        lag = np.searchsorted(t, t - w, side='right') - 1
        ok = lag >= 0
        safe = np.clip(lag, 0, None)
        dt = np.where(ok, t - t[safe], np.nan)
        with np.errstate(invalid='ignore', divide='ignore'):
            blocks.append(np.where(ok[:, None], (X - X[safe]) / dt[:, None], np.nan))
    blocks += [df.cummax().to_numpy(), df.cummin().to_numpy()]
    if USE_RELATIVE:
        blocks += _relative_blocks(t, X)
    return np.concatenate(blocks, axis=1)


def causal_feature_names(signals, windows) -> list[str]:
    names = []
    for w in windows:
        for agg in ('mean', 'max', 'min', 'std', 'rate'):
            names += [f'{s}_{agg}{int(w)}s' for s in signals]
    for agg in ('cummax', 'cummin'):
        names += [f'{s}_{agg}' for s in signals]
    if USE_RELATIVE:
        for agg in ('fromstart', 'ratiomax', 'dropmax', 'sincemax', 'span'):
            names += [f'{s}_{agg}' for s in signals]
    return names


def add_causal_features(base: pd.DataFrame, signals: list[str], windows=WINDOWS_S) -> pd.DataFrame:
    names = causal_feature_names(signals, windows)
    out = np.full((len(base), len(names)), np.nan)
    times = base['relative_time_s'].to_numpy(dtype=float)
    values = base[signals].to_numpy(dtype=float)

    events = [(c, op, thr) for c, op, thr in EVENTS if c in base.columns] if USE_EVENTS else []
    ev_names = [f'elapsed_{c}_{"lt" if op == "<" else "ge"}{thr:g}' for c, op, thr in events]
    ev_out = np.full((len(base), len(events)), np.nan)

    for _, idx in base.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(times[idx], kind='mergesort')]
        out[idx] = _causal_one_experiment(times[idx], values[idx], windows)
        for k, (c, op, thr) in enumerate(events):
            ev_out[idx, k] = _first_hit_elapsed(times[idx], base[c].to_numpy(dtype=float)[idx], op, thr)

    frames = [pd.DataFrame(out, index=base.index, columns=names)]
    if events:
        frames.append(pd.DataFrame(ev_out, index=base.index, columns=ev_names))
    return pd.concat(frames, axis=1)


# ---------------------------------------------------------------------------
# 4. 피처 조립
# ---------------------------------------------------------------------------
def numeric(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return frame[cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)


def gas_columns(frame: pd.DataFrame) -> list[str]:
    if not USE_GAS:
        return []
    return sorted(c for c in frame.columns
                  if c.startswith('meas_') and c not in CORE_SENSORS and c not in EXCLUDE_MEAS)


def build_features(frame: pd.DataFrame, image_features: pd.DataFrame) -> pd.DataFrame:
    """frame에 들어 있는 행만으로 피처를 만듭니다."""
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
        base['gas_zero_frac'] = (g == 0).sum(axis=1) / len(gas)
        base['gas_sum'] = g.sum(axis=1, min_count=1)
        base['gas_max'] = g.max(axis=1)
        base['gas_nonzero'] = (g > 0).sum(axis=1)
        derived += ['gas_zero_frac', 'gas_sum', 'gas_max', 'gas_nonzero']

    use_cnn = USE_CNN and set(CNN_COLS) <= set(image_features.columns)
    img_cols = IMAGE_STATS + (CNN_COLS if use_cnn else [])
    img = image_features.reindex(frame['sample_id'].astype(str).values)[img_cols]
    assert img.notna().all(axis=1).all(), '이미지 피처가 없는 샘플이 있습니다.'
    base[img_cols] = img.values

    signals = sensors + derived + IMAGE_WINDOW_SIGNALS + (CNN_WINDOW_SIGNALS if use_cnn else [])
    causal = add_causal_features(base, signals)

    raw = sensors + derived + gas + img_cols
    if USE_TIME:
        raw = ['relative_time_s'] + raw
    parts = [base[raw], causal]
    if USE_META:
        parts.append(numeric(frame, META_COLS))
    return pd.concat(parts, axis=1)


def subsample_stride(frame: pd.DataFrame, stride: int) -> pd.DataFrame:
    """실험마다 시간순으로 stride 칸에 한 행씩 (Test의 10·15초 간격 흉내)."""
    f = frame.sort_values(['experiment_id', 'relative_time_s'], kind='mergesort')
    keep = (f.groupby('experiment_id').cumcount() % stride) == 0
    return f[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. 모델: 다중분류 + 순서형 분해(Frank & Hall)
# ---------------------------------------------------------------------------
def _lgbm(objective: str, class_weight, n_estimators: int, seed: int):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        objective=objective, n_estimators=n_estimators, learning_rate=0.03,
        num_leaves=31, min_child_samples=80, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.4, reg_lambda=5.0, class_weight=class_weight,
        random_state=seed, n_jobs=-1, verbose=-1,
    )


class MultiModel:
    """다중분류 시드 앙상블."""

    def __init__(self, models):
        self.models = models
        self.classes_ = models[0].classes_

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)

    @property
    def booster_(self):
        return self.models[0].booster_


class OrdinalModel:
    """Frank & Hall 순서형 분해: P(y>1..5)를 각각 이진 분류로 학습해 단계 확률로 환원."""

    def __init__(self, models):
        self.models = models          # {k: 이진 모델} , k = 1..5

    def predict_proba(self, X):
        n = len(X)
        gt = np.zeros((n, 6))         # gt[:, k] = P(y > k)
        gt[:, 0] = 1.0
        for k, m in self.models.items():
            gt[:, k] = m.predict_proba(X)[:, 1]
        gt[:, 1:] = np.minimum.accumulate(gt[:, 1:], axis=1)   # 단조 보정
        proba = np.empty((n, 6))
        for k in range(6):
            proba[:, k] = gt[:, k] - (gt[:, k + 1] if k + 1 < 6 else 0.0)
        proba = np.clip(proba, 1e-9, None)
        return proba / proba.sum(1, keepdims=True)


def fit_models(X, y, class_weight, n_estimators, seeds):
    """다중분류 + 순서형 두 모델을 함께 학습합니다."""
    y = np.asarray(y, dtype=int)
    multi = MultiModel([_lgbm('multiclass', class_weight, n_estimators, SEED + k).fit(X, y)
                        for k in range(seeds)])
    ordinal_models = {}
    for k in range(1, 6):
        target = (y > k).astype(int)
        if target.min() == target.max():
            continue
        ordinal_models[k] = _lgbm('binary', 'balanced', n_estimators, SEED).fit(X, target)
    return multi, OrdinalModel(ordinal_models)


def blend(multi_proba: np.ndarray, ord_proba: np.ndarray, w: float) -> np.ndarray:
    return (1.0 - w) * multi_proba + w * ord_proba


def tune_blend(multi_proba, ord_proba, y, grid=np.arange(0.0, 1.01, 0.1)):
    from sklearn.metrics import f1_score
    best_w, best = 0.0, -1.0
    for w in grid:
        s = f1_score(y, blend(multi_proba, ord_proba, w).argmax(1) + 1,
                     labels=LABELS, average='macro', zero_division=0)
        if s > best:
            best_w, best = float(w), s
    return best_w, best


def select_features(model, features, top_n: int) -> list[str]:
    gain = pd.Series(model.booster_.feature_importance('gain'), index=features)
    return gain.sort_values(ascending=False).head(top_n).index.tolist()


# ---------------------------------------------------------------------------
# 6. 과거 방향 전진 필터 (선택)
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
    return {'T': counts / counts.sum(1, keepdims=True), 'pi0': first / first.sum(),
            'prior': prior / prior.sum()}


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


# ---------------------------------------------------------------------------
# 7. 추론
# ---------------------------------------------------------------------------
def predict_proba_bundle(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    Xf = X[bundle['features']]
    return blend(bundle['multi'].predict_proba(Xf), bundle['ordinal'].predict_proba(Xf), bundle['w'])


def predict_labels(bundle: dict, frame: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    proba = predict_proba_bundle(bundle, X)
    filt = bundle.get('filter')
    if filt is None:
        return (proba.argmax(1) + 1).astype(int)
    return causal_filter(proba, frame, filt['stats'], filt['tau'])


def image_table(frame: pd.DataFrame, roots, bundle: dict, model_path, device: str = 'cpu') -> pd.DataFrame:
    paths = resolve_image_paths(frame, roots)
    img = extract_image_features(frame, paths)
    if bundle.get('cnn'):
        import cnn as C
        cnn_dir = Path(model_path).resolve().parent / bundle['cnn']['dir']
        proba = C.predict_paths(paths.tolist(), [cnn_dir / f for f in bundle['cnn']['files']], device=device)
        img = img.join(C.probs_to_frame(proba, frame['sample_id'].astype(str).values))
    return img


def predict_frame(frame: pd.DataFrame, roots, bundle: dict, model_path, device: str = 'cpu') -> pd.Series:
    img = image_table(frame, roots, bundle, model_path, device)
    X = build_features(frame, img)
    return pd.Series(predict_labels(bundle, frame, X), index=frame['sample_id'].astype(str).values)
