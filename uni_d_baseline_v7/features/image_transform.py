"""Deterministic RGB thermal-image preprocessing for training AND inference.

Default: preserve aspect ratio, resize longest side to 160 px, black center pad,
ImageNet mean/std. Returns a contiguous float32 numpy (C,H,W) array.
No hue/color jitter because RGB is a thermal pseudo-color encoding.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_image_path(image_root: str | Path, image_path: str | Path) -> Path:
    """Accept either split root + 'images/x.png' or 'images/' root + path.

    Never references any other split. Explicitly rejects absolute/parent paths.
    """
    root = Path(image_root)
    rel = Path(image_path)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"Unsafe relative image_path: {image_path}")
    candidates = [root / rel]
    if root.name == "images" and rel.parts[0] == "images":
        candidates.append(root / Path(*rel.parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image not found (tried {', '.join(map(str, candidates))})")


@dataclass(frozen=True)
class ThermalImageTransform:
    size: int = 160
    mode: str = "pad"  # 'pad' preserves ratio; 'stretch' is an ablation only
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    pad_rgb: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        if self.size < 16:
            raise ValueError("Image size must be >=16")
        if self.mode not in ("pad", "stretch"):
            raise ValueError("mode must be 'pad' or 'stretch'")
        if len(self.mean) != 3 or len(self.std) != 3 or any(x <= 0 for x in self.std):
            raise ValueError("Expected 3-channel mean and positive std")

    def __call__(self, image: Image.Image) -> np.ndarray:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if self.mode == "stretch":
            transformed = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        else:
            w, h = image.size
            ratio = min(self.size / w, self.size / h)
            new_w = min(self.size, max(1, round(w * ratio)))
            new_h = min(self.size, max(1, round(h * ratio)))
            transformed = Image.new("RGB", (self.size, self.size), self.pad_rgb)
            resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            transformed.paste(resized, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        arr = np.asarray(transformed, dtype=np.float32) / np.float32(255.0)
        arr = (arr - np.asarray(self.mean, dtype=np.float32)) / np.asarray(self.std, dtype=np.float32)
        return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)

    def load(self, image_root: str | Path, image_path: str | Path) -> np.ndarray:
        resolved = resolve_image_path(image_root, image_path)
        with Image.open(resolved) as image:
            return self(image)

    def config(self) -> dict:
        return {"size": self.size, "mode": self.mode, "mean": list(self.mean),
                "std": list(self.std), "pad_rgb": list(self.pad_rgb)}
