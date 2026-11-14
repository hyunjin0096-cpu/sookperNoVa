"""ResNet-18 (ImageNet 사전학습, 고정) 이미지 특징 추출.

- 종횡비를 유지한 Resize + 검은색 Padding 으로 224×224 를 만들고 ImageNet 평균·표준편차로 정규화합니다.
- 분류층을 뗀 512차원 출력을 소수 3자리로 반올림해 장치·배치에 따른 미세한 수치 차이를 없앱니다.
- 이미지 한 장마다 독립 계산이므로 입력 행 구성과 무관하게 같은 이미지는 같은 값이 나옵니다.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

IMG = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _load(path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert('RGB')
        w, h = im.size
        s = IMG / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        im = im.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new('RGB', (IMG, IMG), (0, 0, 0))
        canvas.paste(im, ((IMG - nw) // 2, (IMG - nh) // 2))
        return np.asarray(canvas, dtype=np.uint8)


def extract_embeddings(paths, device: str = 'cpu', batch_size: int = 128) -> np.ndarray:
    import torch
    import torchvision
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    dev = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    net = net.to(dev).eval()
    mean = torch.tensor(MEAN).view(1, 3, 1, 1).to(dev)
    std = torch.tensor(STD).view(1, 3, 1, 1).to(dev)
    out = []
    with torch.no_grad():
        for s in range(0, len(paths), batch_size):
            arr = np.stack([_load(p) for p in paths[s:s + batch_size]])
            x = torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2).float() / 255.0
            out.append(net((x - mean) / std).float().cpu().numpy())
    return np.round(np.concatenate(out).astype(np.float64), 3)
