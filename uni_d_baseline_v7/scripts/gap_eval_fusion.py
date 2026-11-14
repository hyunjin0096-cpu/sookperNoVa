"""gap_eval_fusion.py - 융합 NN의 '시간 간격 민감도' 진단 (OOF 분석, 제출 아님)

각 fold의 검증 실험을 5/10/15초로 솎아낸 뒤(features.gap_view.make_gap_view) 피처를 다시 계산하고,
fold 모델로 예측해서 '전체 이력 예측(fusion_oof.csv)'과 같은 행 기준으로 Macro F1을 비교한다.

사용(레포 루트에서):
  python -u -m scripts.gap_eval_fusion --data-dir "$D" --folds artifacts/cv/folds_exp.csv \
      --model-dir artifacts/fusion_cv --fusion-oof artifacts/fusion_cv/fusion_oof.csv
저장 위치: scripts/gap_eval_fusion.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from cv.common import load_labeled, attach_folds, LABELS
from features.gap_view import make_gap_view
from features.image_transform import ThermalImageTransform
from features.multimodal_dataset import ImageSensorDataset
from features.sensor_preprocess import SensorPreprocessor
from fusion_model import FusionNet


def load_fold_model(wdir: Path, device):
    need = ["model.pt", "config.json", "sensor_preprocess.json"]
    miss = [n for n in need if not (wdir / n).exists()]
    if miss:
        raise FileNotFoundError(f"{wdir}에 {miss} 없음. 내용: {sorted(p.name for p in wdir.iterdir())}")
    cfg = json.loads((wdir / "config.json").read_text(encoding="utf-8"))
    t = cfg["image_transform"]
    tfm = ThermalImageTransform(size=int(t["size"]), mode=str(t["mode"]), mean=tuple(t["mean"]),
                                std=tuple(t["std"]), pad_rgb=tuple(t["pad_rgb"]))
    prep = SensorPreprocessor.load(wdir / "sensor_preprocess.json")
    model = FusionNet(sensor_dim=int(cfg["sensor_dim"]), pretrained_backbone=False)
    model.load_state_dict(torch.load(wdir / "model.pt", map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval(), prep, tfm


@torch.inference_mode()
def predict(model, prep, tfm, frame, device, bs=64, workers=4):
    fr = frame.drop(columns=["label"], errors="ignore").reset_index(drop=True)
    ds = ImageSensorDataset(fr, prep, image_transform=tfm)  # 행별 _image_root 사용
    out = []
    for b in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers):
        out.append(torch.softmax(model(b["image"].to(device), b["sensor"].to(device)).float(), 1).cpu().numpy())
    return np.concatenate(out)


def f1(y, p):
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--folds", default="artifacts/cv/folds_exp.csv")
    ap.add_argument("--model-dir", default="artifacts/fusion_cv")
    ap.add_argument("--fusion-oof", default="artifacts/fusion_cv/fusion_oof.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 7, 99])
    a = ap.parse_args()
    D = Path(a.data_dir).expanduser()
    df = attach_folds(load_labeled(str(D / "train/train.csv"), str(D / "validation/validation.csv")), a.folds)
    if "_image_root" not in df.columns:
        # 학습 행은 <D>/train, 검증 행은 <D>/validation 폴더 기준으로 이미지를 읽는다
        tr_ids = set(pd.read_csv(D / "train/train.csv", usecols=["sample_id"])["sample_id"])
        df["_image_root"] = np.where(df["sample_id"].isin(tr_ids), str(D / "train"), str(D / "validation"))
    oof = pd.read_csv(a.fusion_oof)[["sample_id", "pred_label"]].rename(columns={"pred_label": "pred_full"})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    rows = []
    for fold in sorted(df.fold.unique()):
        model, prep, tfm = load_fold_model(Path(a.model_dir) / f"fold_{int(fold)}", device)
        val = df[df.fold == fold].copy().reset_index(drop=True)
        for seed in a.seeds:
            thin = make_gap_view(val, seed=seed)
            pr = predict(model, prep, tfm, thin, device)
            pair = thin[["sample_id", "label"]].copy()
            pair["pred_thin"] = np.array(LABELS)[pr.argmax(1)]
            pair = pair.merge(oof, on="sample_id", validate="one_to_one")
            pair["fold"], pair["seed"] = int(fold), seed
            rows.append(pair)
            print(f"fold {int(fold)} seed {seed}: 남은 {len(pair):,}행  "
                  f"전체이력 F1={f1(pair.label, pair.pred_full):.4f}  솎은이력 F1={f1(pair.label, pair.pred_thin):.4f}",
                  flush=True)
    allp = pd.concat(rows, ignore_index=True)
    print("\n=== 시드별 풀링 (같은 행 기준) ===")
    for s, g in allp.groupby("seed"):
        print(f"seed {s}: 전체이력 {f1(g.label, g.pred_full):.4f} -> 솎은이력 {f1(g.label, g.pred_thin):.4f}  "
              f"(예측 바뀐 비율 {(g.pred_full != g.pred_thin).mean():.1%})")
    print("진단용입니다. 리더보드 점수 추정이 아닙니다.")
    allp.to_csv(Path(a.model_dir) / "gap_eval_fusion_pairs.csv", index=False)


if __name__ == "__main__":
    main()