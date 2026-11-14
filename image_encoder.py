"""RGB 열화상 -> ImageNet 사전학습 ResNet18 512차원 + 기존 이미지 통계 6개.
같은 이미지의 특징은 현재 이미지 한 장만으로 계산됩니다.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from baseline_features import IMAGE_STATS, HOT_THRESHOLD, resolve_image_paths
from config import IMAGE_SIZE, BATCH_SIZE, SEED


def init_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resize_pad(image, size=IMAGE_SIZE):
    image = image.convert('RGB')
    image.thumbnail((size, size), Image.Resampling.BILINEAR)
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    return ImageOps.expand(image, border=(left, top, size - image.width - left, size - image.height - top), fill=(0, 0, 0))

TRANSFORM = transforms.Compose([
    transforms.Lambda(resize_pad),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageSamples(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        with Image.open(self.paths[i]) as src:
            rgb = src.convert('RGB')
            # 베이스라인과 같은 L 모드 밝기 통계
            gray = np.asarray(rgb.convert('L'), dtype=np.float32) / 255.0
            stats = np.array([
                gray.mean(), gray.std(), gray.max(),
                np.percentile(gray, 95), np.percentile(gray, 99),
                (gray >= HOT_THRESHOLD).mean(),
            ], dtype=np.float32)
            tensor = TRANSFORM(rgb)
        return tensor, torch.from_numpy(stats)


def load_encoder(weights_path=None):
    init_seed()
    if weights_path:
        # 사전학습 가중치를 로컬에 저장해둔 경우에만 사용
        model = resnet18(weights=None)
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
    else:
        try:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception as exc:
            raise RuntimeError('ResNet18 사전학습 가중치를 다운로드하지 못했습니다. 최초 1회 인터넷 연결 후 실행하거나 --weights 경로를 지정하세요.') from exc
    model.fc = nn.Identity()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.to(device), device


def extract(frame, image_roots, encoder, device, batch_size=BATCH_SIZE, blank=False):
    """순서 보존. blank=True면 이미지가 검은색일 때의 특징을 계산(검사용)."""
    if blank:
        # 원본 이미지 크기는 사용하지 않음: 모든 RGB 입력을 검은색으로 대체
        # blank의 통계는 원본 픽셀 크기와 무관하게 모두 0
        dummy = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        dummy = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(dummy)
        with torch.inference_mode():
            single = encoder(dummy.unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
        embeddings = np.repeat(single[None, :], len(frame), axis=0).astype(np.float32)
        stats = np.zeros((len(frame), len(IMAGE_STATS)), dtype=np.float32)
    else:
        paths = resolve_image_paths(frame, image_roots)
        loader = DataLoader(ImageSamples(paths), batch_size=batch_size, shuffle=False, num_workers=0)
        emb_batches, stat_batches = [], []
        with torch.inference_mode():
            for batch_idx, (imgs, image_stats) in enumerate(loader):
                z = encoder(imgs.to(device)).cpu().numpy().astype(np.float32)
                emb_batches.append(z)
                stat_batches.append(image_stats.numpy())
                if (batch_idx + 1) % 50 == 0:
                    print(f'  이미지 {min((batch_idx+1)*batch_size, len(frame)):,} / {len(frame):,}장')
        embeddings = np.vstack(emb_batches)
        stats = np.vstack(stat_batches)
    assert embeddings.shape == (len(frame), 512)
    embed_frame = pd.DataFrame(embeddings, columns=[f'resnet_{k:03d}' for k in range(512)])
    img_stats = pd.DataFrame(stats, columns=IMAGE_STATS, index=pd.Index(frame['sample_id'].to_numpy(), name='sample_id'))
    return embed_frame, img_stats


def export_pretrained_weights(destination):
    """학습과 동일한 torchvision ResNet18 기본 가중치를 로컬에 저장합니다."""
    from torchvision.models import resnet18, ResNet18_Weights
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    torch.save(model.state_dict(), destination)
    return destination
