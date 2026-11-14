"""Frozen DINOv2 ViT-S/14 CLS embeddings (384 dims), cached by split and sample id."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from PIL import Image

MODEL_ID = 'dinov2_vits14'
DIM = 384


def extract_embeddings(frame: pd.DataFrame, paths: pd.Series, cache_dir: Path,
                       split: str, batch_size: int = 32, image_size: int = 224,
                       device: str = 'auto') -> pd.DataFrame:
    import torch
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    assert image_size % 14 == 0, 'DINOv2 ViT-S/14 입력 크기는 14의 배수여야 합니다.'
    ids = frame['sample_id'].astype(str).to_numpy()
    assert len(set(ids)) == len(ids), 'sample_id가 중복되었습니다.'
    # Validate cache identity and image metadata; prevents silently reusing unrelated images.
    path_list = [Path(p) for p in paths]
    signature = hashlib.sha256()
    signature.update(f'{MODEL_ID}/{image_size}/imagenet_norm/cls'.encode())
    for sid, p in zip(ids, path_list):
        s = p.stat()
        signature.update(f'{sid}|{p.resolve()}|{s.st_size}|{s.st_mtime_ns}\n'.encode())
    key = signature.hexdigest()[:16]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = cache_dir / f'dino_{split}_{image_size}_{key}.npz'
    if filename.exists():
        with np.load(filename, allow_pickle=True) as saved:
            if np.array_equal(saved['sample_id'], ids) and saved['emb'].shape == (len(ids), DIM):
                print('DINO 캐시 재사용:', filename, flush=True)
                return pd.DataFrame(saved['emb'], index=pd.Index(ids, name='sample_id'),
                                    columns=[f'dino_{i:03d}' for i in range(DIM)])

    actual_device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    print(f'DINOv2 모델 로딩: {MODEL_ID}, device={actual_device}', flush=True)
    # First run downloads Facebook Research repository and pretrained weights; internet required.
    backbone = torch.hub.load('facebookresearch/dinov2', MODEL_ID, pretrained=True)
    backbone = backbone.to(actual_device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    embeddings = np.empty((len(ids), DIM), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            images = []
            for p in path_list[start:start + batch_size]:
                with Image.open(p) as im:
                    images.append(transform(im.convert('RGB')))
            x = torch.stack(images).to(actual_device)
            feats = backbone.forward_features(x)['x_norm_clstoken']
            embeddings[start:start + len(images)] = feats.detach().float().cpu().numpy()
            if start == 0 or (start // batch_size + 1) % 50 == 0:
                print(f'  {split}: {min(start + batch_size, len(ids)):,}/{len(ids):,}', flush=True)
    np.savez_compressed(filename, sample_id=ids, emb=embeddings)
    return pd.DataFrame(embeddings, index=pd.Index(ids, name='sample_id'),
                        columns=[f'dino_{i:03d}' for i in range(DIM)])
