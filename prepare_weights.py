"""기존 artifacts 폴더의 학습된 모델을 제출용 weights 폴더로 복사합니다. 재학습하지 않습니다."""
from pathlib import Path
import shutil
from config import ROOT, WEIGHTS_DIR
from image_encoder import export_pretrained_weights

def main():
    old = ROOT / 'artifacts'
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ('lightgbm.joblib', 'metadata.json'):
        src = old / name
        if not src.exists():
            if not (WEIGHTS_DIR / name).exists():
                raise FileNotFoundError(f'{src} 파일이 없습니다. 기존 서버 학습 결과가 필요합니다.')
        else:
            shutil.copy2(src, WEIGHTS_DIR / name)
            print('복사:', name)
    dest = WEIGHTS_DIR / 'resnet18.pth'
    if not dest.exists():
        export_pretrained_weights(dest)
        print('ResNet18 가중치 저장:', dest)
    else:
        print('ResNet18 가중치 존재:', dest)
    print('준비 완료:', WEIGHTS_DIR)

if __name__ == '__main__':
    main()
