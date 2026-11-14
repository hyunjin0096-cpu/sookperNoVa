# ThermalFusion-GBM 제출 코드 (ResNet18 + LightGBM)

## 필수 실행 명령

```bash
python test.py --input /home/ubuntu/data/battery_data_light/test/test.csv --image_root /home/ubuntu/data/battery_data_light --output submission.csv
```

`test.py`는 `weights/lightgbm.joblib`, `weights/metadata.json`, `weights/resnet18.pth`를 읽어 **학습 없이** 입력 CSV의 행만 예측합니다. `--image_root`는 `image_path`가 상대 경로인 이미지 폴더의 기준 경로입니다.

## 이미 학습한 기존 모델을 그대로 이전하기

기존 `artifacts/lightgbm.joblib`, `artifacts/metadata.json`을 이 폴더에 유지한 채:

```bash
python prepare_weights.py
```

두 파일을 `weights/`로 복사하고 ImageNet 사전학습 ResNet18 기본 가중치를 `weights/resnet18.pth`로 저장합니다. 학습을 다시 하지 않습니다. **기존 학습 때 기본 torchvision 사전학습 가중치를 사용한 경우에만 정확히 같은 이미지 인코더를 복원합니다.** 학습 때 `train.py --weights PATH`를 썼다면 그때 사용한 ResNet18 state_dict를 `weights/resnet18.pth`로 복사해야 합니다. 기본 가중치를 새로 받으면 예측이 달라질 수 있습니다.

`resnet18.pth`는 25MB를 넘을 가능성이 높습니다. 공개 레포에 올리기 전 가이드대로 파일을 별도 공유하고 README에 **실제 가중치 파일 다운로드 주소/설치 위치**를 추가하세요. 현재 프로젝트에는 사용자 계정용 링크가 없습니다. 이 파일 없이는 오프라인 평가 서버에서 실행할 수 없습니다.

## 환경

운영체제: 개발 환경 Ubuntu Linux, Python: 서버에서 실제 사용한 버전(`python --version`), GPU: CUDA를 사용할 수 있으면 자동 사용. 학습 당시 사용 패키지 버전을 `pip freeze`로 별도 기록하는 것을 권장합니다.

```bash
pip install -r requirements.txt
python --version
nvidia-smi
```

PyTorch는 requirements.txt에 버전 고정하지 않습니다. torchvision/LightGBM/Python의 호환성을 확인하세요.

## 모델 및 학습/검증

- ImageNet 사전학습 ResNet18 (fine-tuning 없음), 512차원 임베딩
- RGB 이미지 비율 유지 축소 + 224x224 검은 패딩 + ImageNet 정규화
- 기본 이미지 밝기 통계 6개와 센서/실험 조건/과거 시계열 특징을 합쳐 LightGBM 다중 클래스 분류
- 같은 experiment_id 내 **현재 및 과거 행**만 사용해 10초/30초 통계 생성. 미래 시점 사용 금지
- Train 78개 독립 실험, 실험 단위 GroupKFold 5-Fold CV, 별도 Validation 19개 실험. 마지막 학습은 Train만 사용
- 결과(기존 서버 실행): CV Macro F1 0.8878 ± 0.0326, Validation Macro F1 0.7523. 실제 가중치는 사용자 서버에 있음
- 각 클래스 1~6 중 최대 확률 클래스를 선택. 별도 확률 임계값 조정 없음. 밝기 고온 비율 임계값 0.8은 기존 베이스라인 유지

다시 학습할 때만 `python train.py`를 사용하며, 완료되면 `weights/`에 모델/설정/ResNet18 가중치를 저장합니다. 학습 시 데이터 폴더는 `config.py`의 `DATA_DIR`로 지정합니다.

## 제출 검사

```bash
python check_submission.py --data_dir /home/ubuntu/data/battery_data_light --final submission.csv
```

**위 check_submission.py는 운영진 배포 부록에서 별도로 가져와 저장해야 합니다.** 제공된 ZIP에는 들어 있지 않습니다. 행 순서 변경, 실험 분리, 미래 행 제거, 두 번 실행, 제출 결과 재현이 모두 통과해야 합니다. `test.py`는 입력 CSV 내 행만 사용하며 이미지 특징은 현재 샘플 한 장에서 계산합니다.

`--check`는 프로젝트 내부 간이 검사로 공식 검사와 다릅니다. 예: `python test.py --input ... --image_root ... --output submission.csv --check`.

**주의:** 기존 베이스라인과 동일하게 과거 시계열 관측을 사용하는 방식이므로 행 순서를 바꿔도 동일한 `sample_id`의 예측이어야 합니다. 미래 관측 제거 시 남는 이전 행도 동일해야 합니다. 검증 서버에서 일부 행만 임의 삭제하면 시계열 피처가 달라질 수 있으므로 규정에서 요구하는 절차대로 검사하세요.

## 원본 데이터 공개 금지

대회 원본 이미지, CSV, 이미지/센서 캐시와 개인 서버 경로는 public 레포에 올리지 마세요. 제출에는 코드, 요구 라이브러리, 가중치 다운로드 방법을 포함합니다.
