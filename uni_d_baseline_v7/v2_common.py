"""v2_common.py - Uni-D v2 공통 모듈 (torch 없이 동작). 학습 스크립트와 test.py가 같은 함수를 쓴다.

구성
  이미지 입출력   : resolve_image / load_u8 / build_cache
  시간 처리(인과)  : exp_groups / prev_index / window_mean   (모두 같은 실험의 '현재 이하' 시점만 사용)
  피처            : extra_sensor_feats / stack_img_feats / build_feature_table
  GBDT 래퍼       : fit_gb / predict_gb / save_models / load_models (LightGBM, 없으면 sklearn HGB)
  후처리          : causal_filter (전방 필터, 기본 꺼짐)
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

VERSION = "unid-v2.0"
LABELS = [1, 2, 3, 4, 5, 6]
GAS = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf",
       "meas_n2o", "meas_no", "meas_no2", "meas_so2"]
FORBIDDEN = {"label", "experiment_id", "fold", "sample_id", "image_path", "_image_root"}
STACK_WINS = (30, 60)          # 초
GAP_CHOICES_S = (5, 10, 15)    # 학습 증강용 이전 프레임 간격(초). Test에서 가져온 값이 아니라 대회 설명의 고정값
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------ 이미지 입출력
def resolve_image(root, rel):
    root, rel = Path(root), str(rel)
    for c in (root / rel, root / Path(rel).name, root / "images" / Path(rel).name):
        if c.is_file():
            return c
    raise FileNotFoundError(f"이미지를 찾을 수 없음: {root / rel}")


def load_u8(path, size, mode="resize"):
    """PNG -> (size,size,3) uint8. 학습 캐시와 test.py가 반드시 같은 함수를 쓴다."""
    im = Image.open(path).convert("RGB")
    if mode == "resize":
        return np.asarray(im.resize((size, size), Image.BILINEAR))
    w, h = im.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    small = np.asarray(im.resize((nw, nh), Image.BILINEAR))
    a = np.asarray(im)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]], axis=0)
    fill = np.median(border, axis=0).astype(np.uint8)
    canvas = np.empty((size, size, 3), np.uint8)
    canvas[:] = fill
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small
    return canvas


def build_cache(df, size, mode, cache_dir, workers=8):
    """라벨 데이터(df 행 순서) 이미지를 uint8 배열로 캐시. 학습 전용(서버 안에서만 사용, test.py는 읽지 않음)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"v2_lab_{mode}{size}.npy"
    ids = cache_dir / f"v2_lab_{mode}{size}_ids.csv"
    if f.exists() and ids.exists() and pd.read_csv(ids)["sample_id"].tolist() == df["sample_id"].tolist():
        print(f"[cache] 재사용: {f}", flush=True)
        return np.load(f)
    paths = [resolve_image(r, p) for r, p in zip(df["_image_root"], df["image_path"])]
    arr = np.empty((len(df), size, size, 3), np.uint8)
    with ThreadPoolExecutor(workers) as ex:
        for s in range(0, len(paths), 4000):
            res = list(ex.map(lambda p: load_u8(p, size, mode), paths[s:s + 4000]))
            arr[s:s + len(res)] = np.stack(res)
            print(f"[cache] {min(s + 4000, len(paths)):,}/{len(paths):,}", flush=True)
    np.save(f, arr)
    df[["sample_id"]].to_csv(ids, index=False)
    return arr


# ------------------------------------------------------------------ 시간 처리 (인과)
def exp_groups(df):
    """실험별 행 위치 배열 리스트. 각 배열은 (relative_time_s, sample_id) 순서 -> 입력 행 순서와 무관."""
    d = df[["experiment_id", "relative_time_s", "sample_id"]].reset_index(drop=True)
    order = d.sort_values(["experiment_id", "relative_time_s", "sample_id"], kind="stable").index.to_numpy()
    if len(order) == 0:
        return []
    e = d["experiment_id"].to_numpy()[order]
    cuts = np.flatnonzero(e[1:] != e[:-1]) + 1
    return np.split(order, cuts)


def prev_index(df, gap_s=None):
    """각 행의 이전 프레임 위치(없으면 -1). gap_s=None: 같은 실험의 바로 앞 행. 숫자: t-gap_s 이하 마지막 행."""
    out = np.full(len(df), -1, dtype=np.int64)
    t_all = df["relative_time_s"].to_numpy(float)
    for pos in exp_groups(df):
        if gap_s is None:
            out[pos[1:]] = pos[:-1]
        else:
            t = t_all[pos]
            j = np.searchsorted(t, t - gap_s, side="right") - 1
            ok = j >= 0
            out[pos[ok]] = pos[j[ok]]
    return out


def window_mean(t, v, w):
    """시점 t 기준 (t-w, t] 구간의 과거 평균 (현재 포함, 미래 없음). t는 오름차순, v: (n,k)."""
    cs = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, axis=0)])
    lo = np.searchsorted(t, t - w, side="right")
    hi = np.arange(len(t)) + 1
    return (cs[hi] - cs[lo]) / (hi - lo)[:, None]


# ------------------------------------------------------------------ 피처
EXTRA_COLS = ["x_gas_sum", "x_gas_npos", "x_gas_nobs", "x_gas_ever", "x_gas_cmax", "x_gas_ratio_cmax",
              "x_thermal_sat", "x_thermal_cmax", "x_thermal_drop"]


def extra_sensor_feats(df):
    """가스/열화상 온도 보조 피처 (모두 같은 실험의 현재 이하 시점만 사용). df 행 순서로 반환."""
    n = len(df)
    out = {c: np.full(n, np.nan) for c in EXTRA_COLS}
    gas_cols = [c for c in GAS if c in df.columns]
    G = df[gas_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float) if gas_cols else np.zeros((n, 0))
    th = pd.to_numeric(df["thermal_t"], errors="coerce").to_numpy(float) if "thermal_t" in df.columns else np.full(n, np.nan)
    with np.errstate(invalid="ignore"):
        for pos in exp_groups(df):
            g = G[pos]
            s = np.nansum(g, axis=1) if g.shape[1] else np.zeros(len(pos))
            out["x_gas_sum"][pos] = s
            out["x_gas_npos"][pos] = (g > 0).sum(1) if g.shape[1] else 0
            out["x_gas_nobs"][pos] = np.isfinite(g).sum(1) if g.shape[1] else 0
            cmax = np.maximum.accumulate(s)
            out["x_gas_cmax"][pos] = cmax
            out["x_gas_ever"][pos] = np.maximum.accumulate((s > 0).astype(float))
            out["x_gas_ratio_cmax"][pos] = np.where(cmax > 0, s / np.where(cmax > 0, cmax, 1.0), np.nan)
            x = th[pos]
            out["x_thermal_sat"][pos] = (x >= 510).astype(float)
            tc = np.fmax.accumulate(x)
            out["x_thermal_cmax"][pos] = tc
            out["x_thermal_drop"][pos] = tc - x
    return pd.DataFrame(out)


def stack_img_feats(df, P):
    """이미지 모델 확률 P(n,6)를 같은 실험의 과거 이력으로 요약한 피처 (인과)."""
    n = len(df)
    E = P @ np.arange(1, 7)
    base = np.column_stack([P, E])
    names = [f"ip{k}" for k in LABELS] + ["ie"]
    res = {nm: base[:, j].copy() for j, nm in enumerate(names)}
    for w in STACK_WINS:
        for nm in names:
            res[f"{nm}_m{w}"] = np.full(n, np.nan)
    for nm in ("ip4", "ip5", "ip6", "ie"):
        res[f"{nm}_cmax"] = np.full(n, np.nan)
    res["ie_slope30"] = np.full(n, np.nan)
    t_all = df["relative_time_s"].to_numpy(float)
    for pos in exp_groups(df):
        t, v = t_all[pos], base[pos]
        for w in STACK_WINS:
            m = window_mean(t, v, w)
            for j, nm in enumerate(names):
                res[f"{nm}_m{w}"][pos] = m[:, j]
        cm = np.maximum.accumulate(v[:, [3, 4, 5, 6]], axis=0)
        for j, nm in enumerate(("ip4", "ip5", "ip6", "ie")):
            res[f"{nm}_cmax"][pos] = cm[:, j]
        j = np.searchsorted(t, t - 30, side="right") - 1
        ok = j >= 0
        jj = np.where(ok, j, 0)
        dt = t - t[jj]
        res["ie_slope30"][pos] = np.where(ok & (dt > 0), (v[:, 6] - v[jj, 6]) / np.where(dt > 0, dt, 1.0), np.nan)
    return pd.DataFrame(res)


VARIANTS = {  # 이름: (extra 사용, 이미지 확률 피처 사용)
    "A": (False, False),   # 센서 기본 (참고용: 이미지 없음)
    "B": (True, False),    # 센서 기본 + 보조 (참고용: 이미지 없음)
    "C": (True, True),     # 센서 기본 + 보조 + 이미지 확률 요약  <- 기본 최종 후보
    "D": (False, True),    # 센서 기본 + 이미지 확률 요약
}


def build_feature_table(df, P, variant):
    """df: 입력 프레임(라벨 있어도 됨), P: 이미지 모델 확률(n,6) 또는 None. 학습과 test.py가 동일하게 호출."""
    from features.causal_sensor_features import make_causal_sensor_features  # 레포의 검증된 인과 피처
    use_extra, use_img = VARIANTS[variant]
    df = df.reset_index(drop=True)
    base = make_causal_sensor_features(df)
    if len(base) != len(df):
        raise RuntimeError("make_causal_sensor_features 행 수가 입력과 다릅니다")
    if "sample_id" in base.columns:
        if not base["sample_id"].reset_index(drop=True).equals(df["sample_id"]):
            raise RuntimeError("make_causal_sensor_features 출력이 입력 행 순서와 다릅니다")
        base = base.drop(columns="sample_id")
    bad = FORBIDDEN & set(base.columns)
    if bad:
        raise RuntimeError(f"모델 입력에 금지된 열이 있습니다: {bad}")
    parts = [base.reset_index(drop=True)]
    if use_extra:
        parts.append(extra_sensor_feats(df))
    if use_img:
        if P is None:
            raise ValueError("이 variant는 이미지 확률이 필요합니다")
        parts.append(stack_img_feats(df, P))
    X = pd.concat(parts, axis=1)
    return X.apply(pd.to_numeric, errors="coerce").astype("float64")


# ------------------------------------------------------------------ GBDT 래퍼
try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:  # 샌드박스/서버에 lightgbm이 없을 때만 sklearn으로 대체
    lgb = None
    HAVE_LGB = False


def balanced_weights(y):
    y = np.asarray(y)
    cnt = np.array([(y == k).sum() for k in LABELS], float)
    w = len(y) / (6 * np.maximum(cnt, 1))
    return w[y - 1]


def fit_gb(X, y, rounds, seed, w=None):
    y = np.asarray(y)
    w = balanced_weights(y) if w is None else w
    if HAVE_LGB:
        params = dict(objective="multiclass", num_class=6, learning_rate=0.05, num_leaves=31,
                      min_data_in_leaf=60, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=5.0, seed=int(seed), deterministic=True, force_row_wise=True,
                      num_threads=4, verbosity=-1)
        ds = lgb.Dataset(X.to_numpy(float), label=y - 1, weight=w)
        return ("lgb", lgb.train(params, ds, num_boost_round=int(rounds)))
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=int(rounds), learning_rate=0.1, max_leaf_nodes=31,
                                       l2_regularization=1.0, early_stopping=False, random_state=int(seed))
    m.fit(X.to_numpy(float), y, sample_weight=w)
    return ("hgb", m)


def predict_gb(model, X):
    kind, m = model
    if kind == "lgb":
        return np.asarray(m.predict(X.to_numpy(float)), float).reshape(len(X), 6)
    pr = m.predict_proba(X.to_numpy(float))
    out = np.zeros((len(X), 6))
    for k, c in enumerate(m.classes_):
        out[:, int(c) - 1] = pr[:, k]
    return out


def save_models(models, wdir):
    wdir = Path(wdir)
    wdir.mkdir(parents=True, exist_ok=True)
    kind = models[0][0]
    for i, (k, m) in enumerate(models):
        if k == "lgb":
            m.save_model(str(wdir / f"stack_{i}.txt"))
        else:
            import joblib
            joblib.dump(m, wdir / f"stack_{i}.joblib")
    return kind


def load_models(wdir, backend, n):
    wdir = Path(wdir)
    out = []
    for i in range(n):
        if backend == "lgb":
            if not HAVE_LGB:
                raise RuntimeError("가중치는 lightgbm으로 저장되었지만 lightgbm이 설치되어 있지 않습니다")
            out.append(("lgb", lgb.Booster(model_file=str(wdir / f"stack_{i}.txt"))))
        else:
            import joblib
            out.append(("hgb", joblib.load(wdir / f"stack_{i}.joblib")))
    return out


def predict_models(models, X):
    return np.mean([predict_gb(m, X) for m in models], axis=0)


# ------------------------------------------------------------------ 후처리: 인과 전방 필터 (기본 꺼짐)
def _trans(dt, lam, skip):
    s = np.exp(-lam * max(dt, 0.0))
    T = np.zeros((6, 6))
    for i in range(6):
        if i == 5:
            T[5, 5] = 1.0
        elif i == 4:
            T[4, 4], T[4, 5] = s, 1 - s
        else:
            T[i, i], T[i, i + 1], T[i, i + 2] = s, (1 - s) * (1 - skip), (1 - s) * skip
    return T


def causal_filter(df, P, lam, gamma, skip, prior):
    """같은 실험의 과거 행 확률만 합쳐 단계가 거꾸로 튀는 것을 막는다. (운영진 허용 확인 후에만 사용)"""
    prior = np.asarray(prior, float)
    out = np.zeros_like(P)
    t_all = df["relative_time_s"].to_numpy(float)
    for pos in exp_groups(df):
        alpha, prev_t = None, None
        for i in pos:
            e = (np.clip(P[i], 1e-6, 1.0) / prior) ** gamma
            a = e if alpha is None else (alpha @ _trans(t_all[i] - prev_t, lam, skip)) * e
            s = a.sum()
            alpha = a / s if s > 0 else e / e.sum()
            prev_t = t_all[i]
            out[i] = alpha
    return out


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
