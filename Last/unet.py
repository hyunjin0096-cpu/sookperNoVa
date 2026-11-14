"""v4: 임시 마스크(pseudo mask)로 학습한 U-Net 분할 피처.

정답 마스크가 없으므로, 열 지수(model._heat_index, 고정 규칙)로 만든 임시 마스크 2종을 U-Net이 따라 그리도록 학습합니다.
  - 채널 0 warm: 주변보다 따뜻한 영역 (열 지수 >= 0.4, 청록 이상)
  - 채널 1 hot : 고온 영역            (열 지수 >= 0.75, 노랑·빨강·흰색)
규칙 마스크는 픽셀 단위라 잡음이 많고, 표시용 글자·눈금도 섞입니다. U-Net은 주변 맥락을 보고
매끈하고 연결된 영역으로 그려 주므로, 그 예측 마스크에서 '모양' 피처(면적 비율, 덩어리 수, 퍼짐 정도,
중심 위치, 경계 기울기)를 뽑아 GBDT에 넣습니다.

규정 관련
- 라벨은 전혀 쓰지 않으며 Train 이미지로만 학습합니다(Validation·Test 이미지는 학습에 사용하지 않음).
- 사전학습 가중치 없이 처음부터 학습합니다.
- 이미지마다 독립 계산 → 다른 행·미래 정보 사용 없음.
- 모든 이미지를 128×128로 맞추고 비율 피처만 쓰므로 원본 이미지 크기(실험 식별 지름길)는 쓰지 않습니다.
- 피처 추출은 기본적으로 CPU에서 해서 실행마다 같은 값이 나오게 합니다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

import model as M

SIZE = 128
WARM_T, HOT_T = 0.4, 0.75            # 임시 마스크 임계값(고정, 데이터로 학습하지 않음)
UNET_STATS = M.UNET_STATS


# ---------------------------------------------------------------------------
# 1. 입력·임시 마스크
# ---------------------------------------------------------------------------
def load_sample(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """반환: rgb (3,S,S) 0~1, heat (S,S) 0~1, pseudo (2,S,S) {0,1}"""
    with Image.open(path) as img:
        rgb = img.convert('RGB')
    heat, _, _ = M._heat_index(rgb)                       # 원본 크기에서 열 지수 계산(색 섞임 방지)
    rgb_s = np.asarray(rgb.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32).transpose(2, 0, 1) / 255.0
    heat_s = np.asarray(Image.fromarray(heat).resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32)
    pseudo = np.stack([pseudo_mask(heat_s, WARM_T), pseudo_mask(heat_s, HOT_T)]).astype(np.float32)
    return rgb_s, heat_s, pseudo


def pseudo_mask(heat: np.ndarray, thr: float) -> np.ndarray:
    """규칙 기반 임시 마스크: 임계값 → 3×3 열림 연산(점 잡음 제거) → 4픽셀 미만 덩어리 제거."""
    m = ndimage.binary_opening(heat >= thr, structure=np.ones((3, 3)))
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.flatnonzero(sizes >= 4))
    return m


# ---------------------------------------------------------------------------
# 2. 작은 U-Net (채널 16-32-64-128, 약 0.5M 파라미터)
# ---------------------------------------------------------------------------
def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, cin=3, cout=2, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([_block(cin, c[0]), _block(c[0], c[1]), _block(c[1], c[2])])
        self.mid = _block(c[2], c[3])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[3], c[2], 2, 2), nn.ConvTranspose2d(c[2], c[1], 2, 2),
                                 nn.ConvTranspose2d(c[1], c[0], 2, 2)])
        self.dec = nn.ModuleList([_block(c[3], c[2]), _block(c[2], c[1]), _block(c[1], c[0])])
        self.head = nn.Conv2d(c[0], cout, 1)

    def forward(self, x):
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def dice_bce_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1)
    return bce + dice.mean()


# ---------------------------------------------------------------------------
# 3. 학습 (train_unet.py에서 호출)
# ---------------------------------------------------------------------------
class PseudoMaskDataset(torch.utils.data.Dataset):
    def __init__(self, paths, augment=False):
        self.paths, self.augment = list(paths), augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rgb, _, pseudo = load_sample(self.paths[i])
        if self.augment:                                  # 좌우·상하 뒤집기만(색은 건드리지 않음)
            if np.random.rand() < 0.5:
                rgb, pseudo = rgb[:, :, ::-1], pseudo[:, :, ::-1]
            if np.random.rand() < 0.5:
                rgb, pseudo = rgb[:, ::-1, :], pseudo[:, ::-1, :]
        return torch.from_numpy(rgb.copy()), torch.from_numpy(pseudo.copy())


def set_seed(seed=M.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def iou(pred, target):
    inter = (pred & target).sum(dim=(0, 2, 3)).float()
    union = (pred | target).sum(dim=(0, 2, 3)).float()
    return (inter / union.clamp(min=1)).tolist()


# ---------------------------------------------------------------------------
# 4. 예측 마스크 → 모양 피처 (샘플마다 독립)
# ---------------------------------------------------------------------------
def _shape_stats(p_warm, p_hot, heat, pseudo):
    eps = 1e-6
    yy, xx = np.mgrid[0:SIZE, 0:SIZE] / (SIZE - 1)       # 0~1 정규화 좌표
    sw, sh = p_warm.sum(), p_hot.sum()
    warm_bin, hot_bin = p_warm >= 0.5, p_hot >= 0.5
    lab, n = ndimage.label(hot_bin)
    if n:
        sizes = ndimage.sum(hot_bin, lab, range(1, n + 1))
        largest = sizes.max() / sizes.sum()
    else:
        largest = np.nan
    if sh > eps and sw > eps:
        cy_h, cx_h = (yy * p_hot).sum() / sh, (xx * p_hot).sum() / sh
        cy_w, cx_w = (yy * p_warm).sum() / sw, (xx * p_warm).sum() / sw
        spread = np.sqrt(((yy - cy_h) ** 2 * p_hot).sum() / sh + ((xx - cx_h) ** 2 * p_hot).sum() / sh)
        offset = np.hypot(cy_h - cy_w, cx_h - cx_w)
    else:
        spread = offset = np.nan
    gy, gx = np.gradient(heat)
    edge = warm_bin ^ ndimage.binary_erosion(warm_bin)    # warm 영역 경계 픽셀
    disagree = np.mean([(warm_bin != pseudo[0].astype(bool)).mean(), (hot_bin != pseudo[1].astype(bool)).mean()])
    return [
        p_warm.mean(), p_hot.mean(), sh / (sw + eps),
        (heat * p_warm).sum() / sw if sw > eps else np.nan,
        (heat * p_hot).sum() / sh if sh > eps else np.nan,
        np.percentile(heat[warm_bin], 95) if warm_bin.any() else np.nan,
        float(n), largest, spread, offset,
        np.hypot(gy, gx)[edge].mean() if edge.any() else np.nan,
        disagree,                                          # U-Net이 규칙 마스크를 얼마나 고쳤는가
    ]


@torch.no_grad()
def extract_unet_features(frame: pd.DataFrame, paths: pd.Series, weights: Path,
                          device: str = 'cpu', batch_size: int = 128) -> pd.DataFrame:
    """sample_id 인덱스의 U-Net 모양 피처 표. 기본 CPU(재현성)."""
    torch.set_num_threads(4)                              # 스레드 수 고정(CPU 연산 순서 고정 → 재현성)
    net = UNet()
    net.load_state_dict(torch.load(weights, map_location='cpu'))
    net.to(device).eval()
    rows, paths = [], list(paths)
    for s in range(0, len(paths), batch_size):
        batch = [load_sample(p) for p in paths[s:s + batch_size]]
        x = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
        prob = torch.sigmoid(net(x)).cpu().numpy().astype(np.float64)
        for (rgb, heat, pseudo), pr in zip(batch, prob):
            rows.append(_shape_stats(pr[0], pr[1], heat.astype(np.float64), pseudo))
    return pd.DataFrame(rows, columns=UNET_STATS, index=pd.Index(frame['sample_id'].values, name='sample_id'))
