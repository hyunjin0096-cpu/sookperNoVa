"""v2_net.py - 이미지 모델 (torch). 현재 프레임 + 이전 프레임 차이(6채널)를 입력으로 6단계 확률을 낸다.

  입력 : [현재 프레임(정규화 RGB), 현재 - 이전 프레임] (이전 프레임이 없으면 차이 = 0)
  구조 : ResNet18의 stem~layer3만 사용(약 2.8M 파라미터, fp16 저장 시 약 5.6MB) -> 전역 평균 -> Linear(6)
  의도 : 센서 입력 없이 '이미지만' 보는 모델 -> 이미지 신호가 센서에 가려지지 않게 하고, 2단계(스태킹)에서 센서와 결합
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v2_common import IMG_MEAN, IMG_STD, load_u8, prev_index, resolve_image


class ImgNet(nn.Module):
    def __init__(self, pretrained: bool = False, n_out: int = 6):
        super().__init__()
        import torchvision
        weights = None
        if pretrained:
            try:
                weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                m = torchvision.models.resnet18(weights=weights)
            except Exception as e:  # 오프라인 등으로 사전학습 가중치를 못 받으면 무작위 초기화
                print(f"[경고] 사전학습 가중치를 불러오지 못해 무작위 초기화로 진행합니다: {type(e).__name__}", flush=True)
                m = torchvision.models.resnet18(weights=None)
        else:
            m = torchvision.models.resnet18(weights=None)
        conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv1.weight.zero_()
            conv1.weight[:, :3] = m.conv1.weight  # 현재 프레임 채널은 사전학습 가중치, 차이 채널은 0에서 시작
        self.stem = nn.Sequential(conv1, m.bn1, m.relu, m.maxpool)
        self.layers = nn.Sequential(m.layer1, m.layer2, m.layer3)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(256, n_out)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.head(self.drop(x.mean(dim=(2, 3))))


def _norm(u8, dev):
    """(B,H,W,3) uint8 -> 정규화 float (B,3,H,W)"""
    x = u8.to(dev).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(IMG_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMG_STD, device=dev).view(1, 3, 1, 1)
    return (x - mean) / std


def make_input(cur_u8, prev_u8, dev, aug: bool = False, gen=None):
    c, p = _norm(cur_u8, dev), _norm(prev_u8, dev)
    if aug:  # 현재/이전 프레임에 같은 밝기·대비·이동 적용(차이 신호 보존). 색조 변환 없음
        B = c.shape[0]
        a = 1.0 + (torch.rand(B, 1, 1, 1, device=dev, generator=gen) - 0.5) * 0.3
        b = (torch.rand(B, 1, 1, 1, device=dev, generator=gen) - 0.5) * 0.3
        c, p = c * a + b, p * a + b
        pad = 8
        H, W = c.shape[2], c.shape[3]
        oy = int(torch.randint(0, 2 * pad + 1, (1,), generator=gen, device=dev))
        ox = int(torch.randint(0, 2 * pad + 1, (1,), generator=gen, device=dev))
        c = F.pad(c, (pad,) * 4, mode="reflect")[:, :, oy:oy + H, ox:ox + W]
        p = F.pad(p, (pad,) * 4, mode="reflect")[:, :, oy:oy + H, ox:ox + W]
    return torch.cat([c, c - p], dim=1)


def save_half(model, path):
    sd = {k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu())
          for k, v in model.state_dict().items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, path)


def load_half(model, path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model


@torch.inference_mode()
def predict_rows(model, arr, rows, prev, dev, bs=256):
    """학습 데이터 캐시(arr)에서 rows의 확률을 계산 (OOF용). prev: 각 행의 이전 프레임 위치(-1이면 자기 자신)."""
    model.eval()
    out = np.zeros((len(rows), 6), np.float64)
    for s in range(0, len(rows), bs):
        b = rows[s:s + bs]
        pi = prev[b]
        pi = np.where(pi < 0, b, pi)
        x = make_input(torch.from_numpy(arr[b]), torch.from_numpy(arr[pi]), dev)
        out[s:s + len(b)] = torch.softmax(model(x).float(), 1).cpu().numpy()
    return out


@torch.inference_mode()
def image_probs_for_frame(frame, image_root, weights_dir, device):
    """test.py용: 원본 PNG를 --image_root에서 읽어 fold 모델들의 평균 확률(n,6)을 계산. 배치 크기 1(구성 무관 결정성)."""
    wdir = Path(weights_dir)
    cfg = json.loads((wdir / "img_config.json").read_text(encoding="utf-8"))
    size, mode, folds = int(cfg["size"]), str(cfg["mode"]), list(cfg["folds"])
    frame = frame.reset_index(drop=True)
    prev = prev_index(frame, None)  # 같은 실험의 바로 앞 행(시간순). 미래 행은 쓰지 않음
    arr = np.stack([load_u8(resolve_image(image_root, p), size, mode) for p in frame["image_path"]]) \
        if len(frame) else np.zeros((0, size, size, 3), np.uint8)
    probs = np.zeros((len(frame), 6), np.float64)
    for k in folds:
        model = load_half(ImgNet(pretrained=False), wdir / f"img_fold{k}.pt").to(device).eval()
        for i in range(len(frame)):
            j = prev[i] if prev[i] >= 0 else i
            x = make_input(torch.from_numpy(arr[i:i + 1]), torch.from_numpy(arr[j:j + 1]), device)
            probs[i] += torch.softmax(model(x).float(), 1).cpu().numpy()[0]
    return probs / max(len(folds), 1)
