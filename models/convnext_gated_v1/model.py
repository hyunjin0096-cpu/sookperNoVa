"""Uni-DTHON: causal sensor engineering + ConvNeXt-Tiny gated fusion.

No labels or test-wide statistics enter feature construction. Images are interpreted
as RGB pseudocolor images, not per-pixel calibrated temperatures.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import convnext_tiny

STAGES = tuple(range(1, 7))
GASES = ("meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf",
         "meas_n2o", "meas_no", "meas_no2", "meas_so2")
SIGNALS = ("meas_v", "meas_t1", "meas_t2", "meas_p", "thermal_t",
           "atmp_t", "indu_t") + GASES
HISTORY_SIGNALS = ("meas_v", "meas_t1", "meas_t2", "meas_p", "thermal_t",
                   "meas_co", "meas_co2", "meas_hf")
PEAK_SIGNALS = ("meas_t1", "meas_t2", "thermal_t", "meas_p", "meas_co2")
WINDOWS_SEC = (10, 30, 60, 120)
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def build_sensor_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Only current and preceding *available* observations from the same experiment.

    Invariant under row permutations, experiment-wise splitting and removing future rows.
    No test-fitted normalization, all-experiment maxima, label history or future windows.
    """
    frame = frame.reset_index(drop=True)
    required = {"sample_id", "experiment_id", "relative_time_s"}.union(SIGNALS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required sensor columns: {missing}")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be present and unique")
    if frame["experiment_id"].isna().any():
        raise ValueError("experiment_id must be present")

    n = len(frame)
    exps = frame["experiment_id"].astype(str).to_numpy()
    sample = frame["sample_id"].astype(str).to_numpy()
    time = pd.to_numeric(frame["relative_time_s"], errors="coerce").to_numpy(float)
    if not np.isfinite(time).all():
        raise ValueError("relative_time_s must be finite")
    raw = {s: pd.to_numeric(frame[s], errors="coerce").to_numpy(float) for s in SIGNALS}
    feats: dict[str, np.ndarray] = {}

    def add(name, arr):
        feats[name] = np.asarray(arr, dtype=np.float32)

    add("relative_time_s", time)
    for s in SIGNALS:
        add(f"raw_{s}", raw[s])
    for col, valid in (("bat_class", (1, 2, 3)), ("soc_status", (1, 2)),
                       ("tr_method", (1, 2))):
        x = pd.to_numeric(frame.get(col, pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(float)
        for v in valid:
            add(f"{col}_eq{v}", (x == v).astype(float))
    add("raw_bat_cc", pd.to_numeric(frame.get("bat_cc", pd.Series(np.nan, index=frame.index)), errors="coerce"))
    add("indu_t_missing", ~np.isfinite(raw["indu_t"]))
    add("thermal_t_at_510", np.isfinite(raw["thermal_t"]) & (raw["thermal_t"] >= 509.5))
    add("t1_minus_t2", raw["meas_t1"] - raw["meas_t2"])
    add("t1_minus_ambient", raw["meas_t1"] - raw["atmp_t"])
    add("t2_minus_ambient", raw["meas_t2"] - raw["atmp_t"])
    add("thermal_minus_t1", raw["thermal_t"] - raw["meas_t1"])
    gas_stack = np.column_stack([raw[g] for g in GASES])
    add("gas_zero_count", np.sum(gas_stack == 0, axis=1))
    add("gas_all_zero", np.all(gas_stack == 0, axis=1))
    add("gas_positive_count", np.sum(gas_stack > 0, axis=1))

    def blank():
        return np.full(n, np.nan, dtype=np.float32)

    for s in HISTORY_SIGNALS:
        for w in WINDOWS_SEC:
            for suffix in ("mean", "delta", "slope"):
                feats[f"{s}_{suffix}{w}s"] = blank()
        for suffix in ("from_first", "delta_prev", "slope_prev"):
            feats[f"{s}_{suffix}"] = blank()
    for s in PEAK_SIGNALS:
        for suffix in ("running_peak", "drawdown", "time_since_peak"):
            feats[f"{s}_{suffix}"] = blank()
    feats["elapsed_since_previous"] = blank()
    feats["history_observation_count"] = blank()

    # Group handling uses integer positions, retaining each original input row's output position.
    for positions in frame.groupby("experiment_id", sort=False).indices.values():
        ix = np.asarray(positions, dtype=int)
        ix = ix[np.lexsort((sample[ix], time[ix]))]
        t = time[ix]
        if len(t) > 1 and np.any(np.diff(t) <= 0):
            raise ValueError("Duplicate/non-increasing relative_time_s in an experiment")
        k = len(ix)
        feats["elapsed_since_previous"][ix] = np.r_[np.nan, np.diff(t)]
        feats["history_observation_count"][ix] = np.arange(1, k + 1)
        for s in HISTORY_SIGNALS:
            x = raw[s][ix]
            good = np.isfinite(x)
            prefix = np.r_[0.0, np.cumsum(np.where(good, x, 0.0))]
            prefix_count = np.r_[0, np.cumsum(good.astype(int))]
            first_pos = np.flatnonzero(good)
            if len(first_pos):
                feats[f"{s}_from_first"][ix] = x - x[first_pos[0]]
            if k > 1:
                pair_good = good[1:] & good[:-1]
                prev_delta = np.where(pair_good, x[1:] - x[:-1], np.nan)
                feats[f"{s}_delta_prev"][ix[1:]] = prev_delta
                feats[f"{s}_slope_prev"][ix[1:]] = prev_delta / np.diff(t)
            for w in WINDOWS_SEC:
                # mean of observed timestamps in [t-w,t], inclusive; no future.
                left = np.searchsorted(t, t - w, side="left")
                end = np.arange(1, k + 1)
                counts = prefix_count[end] - prefix_count[left]
                totals = prefix[end] - prefix[left]
                feats[f"{s}_mean{w}s"][ix] = np.divide(
                    totals, counts, out=np.full(k, np.nan), where=counts > 0)
                # At-or-before t-w comparison. No invented samples at missing timestamps.
                lag = np.searchsorted(t, t - w, side="right") - 1
                now = np.flatnonzero((lag >= 0) & good)
                if len(now):
                    past = lag[now]
                    mask = good[past]
                    now, past = now[mask], past[mask]
                    if len(now):
                        d = x[now] - x[past]
                        dt = t[now] - t[past]
                        feats[f"{s}_delta{w}s"][ix[now]] = d
                        feats[f"{s}_slope{w}s"][ix[now]] = d / dt
        for s in PEAK_SIGNALS:
            x = raw[s][ix]
            peak = -np.inf
            peak_t = np.nan
            for j, val in enumerate(x):
                if np.isfinite(val):
                    if val > peak:
                        peak, peak_t = float(val), float(t[j])
                    feats[f"{s}_running_peak"][ix[j]] = peak
                    feats[f"{s}_drawdown"][ix[j]] = peak - val
                    feats[f"{s}_time_since_peak"][ix[j]] = t[j] - peak_t

    out = pd.DataFrame(feats, index=frame.index)
    return out.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def augment_cadences(frame: pd.DataFrame, enabled: bool = True):
    """Training-only 10s/15s observation thinning. Compute features AFTER thinning.

    Returns (features, source_row), where source_row maps to original images/labels.
    """
    orig = frame.reset_index(drop=True).copy()
    orig["__source_row"] = np.arange(len(orig))
    views = [orig]
    if enabled:
        order = orig.sort_values(["experiment_id", "relative_time_s", "sample_id"], kind="mergesort")
        pos = order.groupby("experiment_id", sort=False).cumcount()
        for stride in (2, 3):
            views.append(order.iloc[np.flatnonzero(pos.to_numpy() % stride == 0)].copy())
    result = []
    for view in views:
        features = build_sensor_features(view)
        features["__source_row"] = view["__source_row"].to_numpy(int)
        result.append(features)
    combined = pd.concat(result, ignore_index=True)
    source_rows = combined.pop("__source_row").to_numpy(dtype=int)
    return combined, source_rows


class SensorNormalizer:
    """Median imputation and z-score normalization; fit only on train fold/train."""
    def __init__(self, columns=None, medians=None, means=None, scales=None):
        self.columns = columns
        self.medians = medians
        self.means = means
        self.scales = scales

    def fit(self, features: pd.DataFrame):
        self.columns = list(features.columns)
        v = features.to_numpy(dtype=np.float64)
        v[~np.isfinite(v)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(v, axis=0)
        med[~np.isfinite(med)] = 0.0
        z = np.where(np.isfinite(v), v, med)
        mean = z.mean(axis=0)
        scale = z.std(axis=0)
        scale[scale < 1e-6] = 1.0
        self.medians, self.means, self.scales = med, mean, scale
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        if list(features.columns) != self.columns:
            raise ValueError("Feature columns/order do not match saved train normalization")
        v = features.to_numpy(dtype=np.float64)
        v[~np.isfinite(v)] = np.nan
        z = np.where(np.isfinite(v), v, self.medians)
        return np.clip((z - self.means) / self.scales, -12.0, 12.0).astype(np.float32)

    def to_dict(self):
        return {"columns": self.columns, "medians": self.medians.tolist(),
                "means": self.means.tolist(), "scales": self.scales.tolist()}

    @classmethod
    def from_dict(cls, state):
        return cls(state["columns"], np.asarray(state["medians"], dtype=float),
                   np.asarray(state["means"], dtype=float),
                   np.asarray(state["scales"], dtype=float))


def make_backbone():
    backbone = convnext_tiny(weights=None)
    backbone.classifier[2] = nn.Identity()  # official ConvNeXt LN -> flatten -> 768D
    return backbone


def load_pretrained_backbone(path: Path | None = None, allow_download: bool = False):
    """No implicit downloads. Accept official Torchvision ImageNet state_dict."""
    model = convnext_tiny(weights=None)
    if path is not None:
        raw = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
            raw = raw["model"]
        if not isinstance(raw, dict):
            raise ValueError("Expected torchvision convnext_tiny state_dict")
        raw = {k.removeprefix("module."): v for k, v in raw.items()}
        model.load_state_dict(raw, strict=True)
    elif allow_download:
        from torchvision.models import ConvNeXt_Tiny_Weights
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Provide --pretrained-weights PATH or --download-pretrained (if rules allow)")
    model.classifier[2] = nn.Identity()
    return model


def image_tensor(path: Path, size: int = 224) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        # Preserve entire small battery crop and aspect ratio; no random hue transforms.
        im = ImageOps.pad(im, (size, size), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
        v = np.asarray(im, dtype=np.float32) / 255.0
    v = (v - np.asarray(IMAGE_MEAN, dtype=np.float32)) / np.asarray(IMAGE_STD, dtype=np.float32)
    return torch.from_numpy(v.transpose(2, 0, 1).copy())


def image_paths(frame: pd.DataFrame, image_root: Path) -> list[Path]:
    root = Path(image_root).resolve()
    paths = []
    for rel in frame["image_path"].astype(str):
        p = Path(rel)
        if p.is_absolute():
            # Accept only paths physically under the supplied root.
            target = p.resolve()
        else:
            target = (root / p).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Image path escapes --image_root: {rel}")
        if not target.is_file():
            raise FileNotFoundError(f"Missing image: {target}")
        paths.append(target)
    return paths


class ImagePathDataset(Dataset):
    def __init__(self, paths, size):
        self.paths = list(paths)
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return image_tensor(self.paths[idx], self.size)


@torch.inference_mode()
def encode_images(backbone, paths, device, batch_size=64, num_workers=2, size=224):
    from torch.utils.data import DataLoader
    backbone.eval().to(device)
    loader = DataLoader(ImagePathDataset(paths, size), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    pieces = []
    for batch in loader:
        img = backbone(batch.to(device, non_blocking=True))
        pieces.append(img.cpu().numpy().astype(np.float32))
    return np.concatenate(pieces, axis=0) if pieces else np.zeros((0, 768), dtype=np.float32)


class GatedFusion(nn.Module):
    """Each modality has an independent head; sample-wise 128D gate blends embeddings."""
    def __init__(self, sensor_dim: int, dropout: float = 0.20):
        super().__init__()
        self.sensor_encoder = nn.Sequential(nn.Linear(sensor_dim, 256), nn.LayerNorm(256),
                                            nn.GELU(), nn.Dropout(dropout),
                                            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU())
        self.image_encoder = nn.Sequential(nn.Linear(768, 256), nn.LayerNorm(256),
                                           nn.GELU(), nn.Dropout(dropout),
                                           nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 128), nn.Sigmoid())
        self.final = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 6))
        self.sensor_head = nn.Linear(128, 6)
        self.image_head = nn.Linear(128, 6)

    def forward(self, sensors, images):
        hs = self.sensor_encoder(sensors)
        hi = self.image_encoder(images)
        gate = self.gate(torch.cat([hs, hi], dim=1))
        fused = gate * hi + (1.0 - gate) * hs
        return self.final(fused), self.sensor_head(hs), self.image_head(hi), gate


def set_determinism(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
