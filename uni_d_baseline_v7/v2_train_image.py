"""v2_train_image.py - 1단계: 이미지 전용 모델을 fold별로 학습해 OOF 확률과 fold 가중치(fp16)를 만든다.

  - 입력: 현재 프레임 + (현재 - 이전 프레임). 학습 때 이전 프레임 간격을 5/10/15초 중 무작위(균등)로 선택 -> Test 간격 불규칙 대비
  - fold는 고정 파일(folds_exp.csv) 사용, 에폭 고정(검증 fold로 조기종료하지 않음)
  - OOF 확률은 '저장한 fp16 가중치를 다시 불러와서' 계산 -> test.py가 보는 값과 동일
  - 산출물: <out>/img_oof.csv, <weights>/img_fold{k}.pt, <weights>/img_config.json

사용(레포 루트):
  python -u -m v2_train_image --data-dir "$D" --folds artifacts/cv/folds_exp.csv --smoke          # 먼저 1~2분 점검
  python -u -m v2_train_image --data-dir "$D" --folds artifacts/cv/folds_exp.csv 2>&1 | tee v2_img.log
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from cv.common import attach_folds, load_labeled
from v2_common import GAP_CHOICES_S, LABELS, VERSION, build_cache, prev_index, write_json
from v2_net import ImgNet, load_half, make_input, predict_rows, save_half


def class_weights(y):
    cnt = np.array([(y == k).sum() for k in LABELS], float)
    w = np.sqrt(cnt.sum() / (6 * np.maximum(cnt, 1)))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--folds", default="artifacts/cv/folds_exp.csv")
    ap.add_argument("--out-dir", default="artifacts/v2")
    ap.add_argument("--weights-dir", default="weights_v2")
    ap.add_argument("--cache-dir", default="cache_v2")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--mode", choices=["resize", "pad"], default="resize")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--fold-ids", type=int, nargs="+", default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="아주 작은 데이터로 1에폭만 (동작 점검용)")
    a = ap.parse_args()

    D = Path(a.data_dir).expanduser()
    df = attach_folds(load_labeled(str(D / "train/train.csv"), str(D / "validation/validation.csv")), a.folds)
    df = df.reset_index(drop=True)
    df["fold"] = df["fold"].astype(int)
    tr_ids = set(pd.read_csv(D / "train/train.csv", usecols=["sample_id"])["sample_id"])
    df["_image_root"] = np.where(df["sample_id"].isin(tr_ids), str(D / "train"), str(D / "validation"))
    y = df["label"].to_numpy(int)
    folds = [int(k) for k in sorted(df["fold"].unique())] if a.fold_ids is None else sorted(a.fold_ids)
    epochs = 1 if a.smoke else a.epochs
    if a.smoke:
        folds = folds[:1]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    print(f"device={dev}, rows={len(df):,}, folds={folds}, size={a.size}, mode={a.mode}, epochs={epochs}", flush=True)

    arr = build_cache(df, a.size, a.mode, a.cache_dir)
    prev_gap = np.stack([prev_index(df, g) for g in GAP_CHOICES_S])  # (3, N)
    prev_now = prev_index(df, 5)                                      # 평가용: 5초 전 행(= 바로 앞 행)
    out_dir, wdir = Path(a.out_dir), Path(a.weights_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wdir.mkdir(parents=True, exist_ok=True)

    oof = np.full((len(df), 6), np.nan)
    for k in folds:
        t0 = time.time()
        torch.manual_seed(a.seed + int(k))
        rng = np.random.default_rng(a.seed + int(k))
        tr = np.flatnonzero((df["fold"] != k).to_numpy())
        va = np.flatnonzero((df["fold"] == k).to_numpy())
        if a.smoke:
            tr, va = tr[:768], va[:512]
        model = ImgNet(pretrained=not a.no_pretrained).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        steps = epochs * int(np.ceil(len(tr) / a.batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights(y[tr]).to(dev), label_smoothing=0.05)
        for ep in range(epochs):
            model.train()
            perm = rng.permutation(tr)
            tot, cnt = 0.0, 0
            for s in range(0, len(perm), a.batch_size):
                b = perm[s:s + a.batch_size]
                g = rng.integers(0, len(GAP_CHOICES_S), len(b))
                pi = prev_gap[g, b]
                pi = np.where(pi < 0, b, pi)
                x = make_input(torch.from_numpy(arr[b]), torch.from_numpy(arr[pi]), dev, aug=True)
                yt = torch.from_numpy(y[b] - 1).to(dev)
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                    loss = loss_fn(model(x).float(), yt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss) * len(b)
                cnt += len(b)
            print(f"fold {k} epoch {ep + 1:02d}/{epochs}: loss={tot / cnt:.4f} ({time.time() - t0:.0f}s)", flush=True)
        path = wdir / f"img_fold{k}.pt"
        save_half(model, path)
        reloaded = load_half(ImgNet(pretrained=False), path).to(dev)     # test.py와 같은 방식으로 다시 불러와 OOF 계산
        p = predict_rows(reloaded, arr, va, prev_now, dev)
        oof[va] = p
        f = f1_score(y[va], p.argmax(1) + 1, labels=LABELS, average="macro", zero_division=0)
        print(f"fold {k}: 이미지 단독 OOF Macro F1={f:.4f} (행 {len(va):,})", flush=True)

    done = ~np.isnan(oof[:, 0])
    if done.all() or not a.smoke:
        res = pd.DataFrame(oof, columns=[f"prob_{k}" for k in LABELS])
        res.insert(0, "sample_id", df["sample_id"].to_numpy())
        res.insert(1, "fold", df["fold"].to_numpy())
        res = res[done]
        res.to_csv(out_dir / ("img_oof_smoke.csv" if a.smoke else "img_oof.csv"), index=False)
    if done.any():
        pf = f1_score(y[done], oof[done].argmax(1) + 1, labels=LABELS, average="macro", zero_division=0)
        print(f"[요약] 이미지 단독 풀링 OOF Macro F1={pf:.4f} ({done.sum():,}행)", flush=True)
    write_json(wdir / "img_config.json", dict(version=VERSION, arch="resnet18-l3-6ch", size=a.size, mode=a.mode,
                                              folds=[int(k) for k in folds], gaps=list(GAP_CHOICES_S),
                                              epochs=epochs, seed=a.seed))
    print("FINISHED", flush=True)


if __name__ == "__main__":
    main()
