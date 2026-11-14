"""U-Net(임시 마스크) 학습 스크립트. GPU가 있으면 GPU로 학습합니다.

예) python train_unet.py --data_dir ./data/battery_data_light --out unet.pt

- Train 이미지만 사용(라벨 사용 없음). 실험마다 고르게 최대 --max_images장 뽑아 학습합니다.
- Validation 이미지의 임시 마스크와의 IoU를 출력합니다(따라 그리기가 잘 되는지 확인용, 단계 분류 점수 아님).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import model as M
import unet as U


def sample_paths(data_dir: Path, split: str, max_images: int, seed: int) -> list:
    frame = pd.read_csv(data_dir / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    paths = M.resolve_image_paths(frame, [data_dir / split, data_dir])
    if len(frame) > max_images:                          # 실험별로 같은 비율로 뽑아 특정 실험 편중 방지
        frac = max_images / len(frame)
        idx = frame.groupby('experiment_id', group_keys=False).apply(
            lambda g: g.sample(max(1, int(round(len(g) * frac))), random_state=seed)).index
        paths = paths.loc[idx]
    return paths.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('unet.pt'))
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--max_images', type=int, default=12000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    U.set_seed()
    tr_paths = sample_paths(args.data_dir, 'train', args.max_images, M.SEED)
    va_paths = sample_paths(args.data_dir, 'validation', 2000, M.SEED)
    print(f'U-Net 학습 이미지 {len(tr_paths):,}장 | 확인용 Validation {len(va_paths):,}장 | device={args.device}')
    g = torch.Generator().manual_seed(M.SEED)
    tr_dl = torch.utils.data.DataLoader(U.PseudoMaskDataset(tr_paths, augment=True), batch_size=args.batch_size,
                                        shuffle=True, num_workers=args.workers, generator=g,
                                        worker_init_fn=lambda w: np.random.seed(M.SEED + w))
    va_dl = torch.utils.data.DataLoader(U.PseudoMaskDataset(va_paths), batch_size=args.batch_size,
                                        num_workers=args.workers)

    net = U.UNet().to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr_dl))
    for epoch in range(1, args.epochs + 1):
        t0, losses = time.time(), []
        net.train()
        for x, y in tr_dl:
            x, y = x.to(args.device), y.to(args.device)
            loss = U.dice_bce_loss(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        net.eval()
        with torch.no_grad():
            ious = []
            for x, y in va_dl:
                pred = torch.sigmoid(net(x.to(args.device))) >= 0.5
                ious.append(U.iou(pred.cpu(), y.bool()))
        warm, hot = np.mean([i[0] for i in ious]), np.mean([i[1] for i in ious])
        print(f'epoch {epoch}/{args.epochs} | loss {np.mean(losses):.4f} | Val IoU warm {warm:.3f} hot {hot:.3f} | {time.time() - t0:.0f}s')

    torch.save(net.cpu().state_dict(), args.out)
    print('U-Net 저장:', args.out)


if __name__ == '__main__':
    main()
