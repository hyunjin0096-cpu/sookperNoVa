"""v2_test.py - 대회 제출용 추론 (학습/재fit 없음). 최종 제출 때 test.py로 복사해서 사용.

  python test.py --input TEST_CSV --image_root IMAGE_DIR --output SUBMISSION_CSV

흐름: 원본 PNG(--image_root)를 읽어 이미지 모델(fold 모델 평균)로 확률 계산 -> 센서 인과 피처 + 이미지 확률 요약 -> LightGBM -> (선택) 인과 필터 -> 라벨
가중치는 test.py 옆 weights/ 폴더에서 읽는다: img_fold*.pt, img_config.json, stack_config.json, stack_*.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v2_common import LABELS, build_feature_table, causal_filter, load_models, predict_models


def compute_image_probs(frame, image_root, weights_dir, device):
    import torch
    from v2_net import image_probs_for_frame
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 사용할 수 없습니다")
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return image_probs_for_frame(frame, image_root, weights_dir, dev)


def predict_frame(frame, image_root, weights_dir, device="auto"):
    wdir = Path(weights_dir)
    cfg = json.loads((wdir / "stack_config.json").read_text(encoding="utf-8"))
    if not str(cfg.get("version", "")).startswith("unid-v2"):
        raise ValueError("알 수 없는 모델 형식")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("input CSV의 sample_id는 비어 있거나 중복될 수 없습니다")
    df = frame.drop(columns=["label"], errors="ignore").reset_index(drop=True)   # 라벨은 사용하지 않음
    P = compute_image_probs(df, image_root, wdir, device)
    X = build_feature_table(df, P, cfg["variant"])
    missing = [c for c in cfg["feature_columns"] if c not in X.columns]
    if missing:
        raise ValueError(f"학습 때 있던 피처가 없습니다: {missing[:5]}")
    X = X[cfg["feature_columns"]]
    models = load_models(wdir, cfg["backend"], int(cfg["n_models"]))
    probs = predict_models(models, X)
    w = float(cfg.get("blend_w", 0.0))
    if w > 0:
        probs = (1 - w) * probs + w * P          # 이미지 모델 확률을 일정 비중으로 섞음
    elif cfg["variant"] in ("A", "B"):
        raise ValueError("이 모델은 이미지를 사용하지 않습니다 (제출 불가 구성)")
    f = cfg.get("filter")
    if f:
        probs = causal_filter(df, probs, f["lam"], f["gamma"], f["skip"], np.asarray(f["prior"], float))
    return np.array(LABELS)[probs.argmax(1)].astype(np.int64), probs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--weights-dir", default=None, help="기본값: 이 파일 옆의 weights/")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    wdir = Path(args.weights_dir).resolve() if args.weights_dir else here / "weights_v2"
    frame = pd.read_csv(args.input)
    labels, _ = predict_frame(frame, args.image_root, wdir, args.device)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": frame["sample_id"], "label": labels}).to_csv(out, index=False)
    print(f"Saved {len(labels)} predictions -> {out}", flush=True)


if __name__ == "__main__":
    main()