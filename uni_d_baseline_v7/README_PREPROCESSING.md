# Uni-DTHON 전처리 패키지 v1.1 — 서버 실행 안내

이 패키지는 기존 `features/causal_sensor_features.py` 위에 **새 파일만 추가**합니다. 기존 `model.py`, `train.py`, `test.py`, frozen `folds_exp.csv` 및 모델 가중치를 절대 덮어쓰지 않습니다.

## 파일 구성

```text
features/sensor_preprocess.py     인과적 센서 특징 + Fold별 통계 학습/추론 + JSON 저장
features/image_transform.py      RGB/Resize/Pad/ImageNet 정규화 (160x160)
features/multimodal_dataset.py   PyTorch 배치 연동 (센서/이미지/sample_id/라벨)
tests/test_preprocessing.py      합성 데이터 단위 테스트
scripts/preprocess_smoke.py      대회 서버 이미지 1장과 일부 행으로 동작 확인
```

**중요:** 이 ZIP은 기존 UniD_CV_v1_1을 설치한 프로젝트에 추가하는 패키지입니다. 단독 디렉터리에서 실행하면 `features/causal_sensor_features.py`가 없어 동작하지 않습니다.

## 1. 설치 (GPU 서버)

Windows CMD에서 업로드 (서버 터미널 X):

```cmd
scp "%USERPROFILE%\Downloads\UniD_Preprocessing_v1_1.zip" ubuntu@118.216.110.86:/home/ubuntu/jihyun/
```

서버 터미널:

```bash
source ~/venv/bin/activate
cd ~/jihyun/uni_d_baseline_v7
# 추가 파일을 확인한 뒤 압축 해제
unzip -l ~/jihyun/UniD_Preprocessing_v1_1.zip
python3 -m zipfile -e ~/jihyun/UniD_Preprocessing_v1_1.zip .
```

실험용 폴더나 모델 파일을 지우지 마세요. 기존 `features/__init__.py`를 변경하지 않습니다.

## 2. 기본 테스트

```bash
python -m unittest discover -s tests -v
# 이전 CV 테스트 + 신규 전처리 테스트가 모두 OK인지 확인
```

## 3. 대회 실제 CSV/이미지 경로 테스트 (학습하지 않음)

```bash
D=~/data/battery_data_light
python -m scripts.preprocess_smoke --data-dir "$D"
```

기대 출력 예시:

```text
PREPROCESS SMOKE OK
sensor feature D 149 image shape (3, 160, 160)
shuffle-invariant: OK; train-fitted validation transform: OK
```

이는 일부 Train 행만을 사용한 **기능 테스트**일 뿐 완성된 스케일러나 모델은 아닙니다. 실험용 데이터를 저장하지 않습니다.

## 4. 5-Fold의 실제 학습 코드에 연결할 방법

```python
from cv.common import load_labeled, attach_folds
from features.sensor_preprocess import SensorPreprocessor
from features.image_transform import ThermalImageTransform
from features.multimodal_dataset import ImageSensorDataset

D = "/home/ubuntu/data/battery_data_light"
data = attach_folds(
    load_labeled(f"{D}/train/train.csv", f"{D}/validation/validation.csv"),
    "artifacts/cv/folds_exp.csv"
)
fold = 0
outer_train = data[data.fold != fold].reset_index(drop=True)
outer_valid = data[data.fold == fold].reset_index(drop=True)

# 반드시 Outer Train에서만 fit!
preprocess = SensorPreprocessor(windows_s=(30, 60), clip=10).fit(outer_train)
x_train = preprocess.transform(outer_train)  # float32, N x 149
x_valid = preprocess.transform(outer_valid)  # NEVER fit on valid
print(x_train.shape, x_valid.shape)

# 통합 Train+Validation 이미지의 루트는 서로 다름!
# 다음은 원본 CSV의 sample_id를 사용하여 split 출처를 구분하는 안전한 방법.
import pandas as pd
train_ids = set(pd.read_csv(f"{D}/train/train.csv", usecols=["sample_id"]).sample_id)
for frame in (outer_train, outer_valid):
    frame["_image_root"] = [
        f"{D}/train" if sid in train_ids else f"{D}/validation"
        for sid in frame.sample_id
    ]

image_tfm = ThermalImageTransform(size=160, mode="pad")
train_dataset = ImageSensorDataset(outer_train, preprocess, image_transform=image_tfm)
valid_dataset = ImageSensorDataset(outer_valid, preprocess, image_transform=image_tfm)
# DataLoader(train_dataset, batch_size=64, shuffle=True), ...
```

위 예제의 `fold=0`은 배선 확인용이고, 전체 OOF를 만들 때는 0~4를 전부 학습합니다. 내부 Early Stopping을 쓴다면 **Inner Train으로 다시 preprocessor.fit**해야 합니다.

## 5. 최종 학습 및 추론에서 동일한 통계 사용

```python
# 최종 학습: 97개 라벨 데이터 전부로 한 번 fit
preprocess = SensorPreprocessor().fit(all_labeled_train)
preprocess.save("weights/sensor_preprocess.json")

# test.py: 저장된 통계만 load. fit 절대 금지.
preprocess = SensorPreprocessor.load("weights/sensor_preprocess.json")
X_test = preprocess.transform(test_frame)
```

이미지는 `ThermalImageTransform(size=160, mode="pad")`로 고정하며 설정을 모델 체크포인트/README에도 저장합니다. `test.py` 실행 시 이미지는 `test_frame.image_path`와 CLI `--image_root`를 조합해 읽습니다. `--image_root`가 `.../test` 또는 `.../test/images`이어도 이미지 경로를 찾아냅니다.

## 6. 꼭 지킬 것

- `experiment_id`별 현재·과거 **존재하는 관측값**만 특징을 계산합니다. 미래 행 삭제/행 셔플/다른 실험 제거 시 기존 예측은 바뀌지 않아야 합니다.
- `meas_a`, `meas_t1_plus_t2`, `indu_t.1`은 미사용; `meas_co==0` 같은 실제 0은 결측 아님.
- 입력 `bat_class`/`soc_status`/`tr_method`는 학습·추론 공통의 **고정 범주 one-hot**; `bat_cc`/시간은 log1p.
- `17 sensor × (현재·직전 변화율·30초 평균·60초 평균) + 시간2 + 용량1 = 71 numeric`; scaled + missing flag = 142; 범주 one-hot 7개 ⇒ **총 D=149**.
- 이미지 증강 없음; 의사색상 정보를 망가뜨리는 hue/color jitter 금지. `stretch` 모드는 **ablation 실험용**으로만 사용합니다.
- 최종 제출 `test.py`는 이 코드만으로 완성되지 않습니다. 모델/체크포인트 로드·예측/CSV 저장을 다음 단계에서 구현합니다. 공식 `check_submission.py` 검사가 별도로 필요합니다.
- 공개 GitHub에는 원본 CSV/이미지/실험 ID가 담긴 Fold 결과물을 올리지 마세요. 최종 필요한 모델 가중치는 운영진 안내에 따라 포함합니다.
