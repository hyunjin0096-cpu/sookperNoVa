# Uni-DTHON · EDA-driven ConvNeXt-Tiny + Sensor MLP + Gated Fusion

> **상태:** 실제 대회 Train 데이터로 학습 및 독립 Validation 평가 완료. GitHub에는 소스코드만 포함하며 학습 가중치는 별도 공유합니다.

## 1. 우리가 실제 EDA에서 확인한 사실 → 설계 반영

| EDA 관찰 / 데이터 구조 | 설계 결정 | 검증할 가설 (아직 증명된 성능 개선 아님) |
|---|---|---|
| Train 33,522행/78실험, Validation 10,200행/19실험, Test 3,451행/11실험 | 실험 ID 기준 5-Fold GroupKFold, Validation은 외부 검증 전용 | 실험 간 일반화 확인 |
| 1·3단계 집중, 6단계 약 2.5~3% | 클래스 빈도 제곱근 역수 기반 손실 가중치(0.4~3 클리핑), Macro F1, 4·5·6단계 별도 보고 | 희소 단계의 분류 성능 개선 |
| Train/Validation 5초; Test 5/10/15초 관측 | 모든 시간 창을 실제 초로 계산; Train을 10/15초 관측으로 얇게 만든 *뒤* 특징 재계산 | 희소 관측에도 강건하게 분류 |
| 6단계 내부에서 일부 온도/가스 수치 감소, 5→6도 항상 센서 증가가 아님 | 최고치 대비 하락(drawdown), 직전/과거 변화량·기울기, 최근 10/30/60/120초 평균; 단계 상승 강제 보정 없음 | 냉각·후반 열폭주 구분력 향상 |
| 가열/과충전 방식에 따라 열화상·가스·온도 분포 다름 | `tr_method` one-hot, 조건·배터리 메타데이터, 방법별 Validation F1 출력 | 서로 다른 실험 조건의 이질성 반영 |
| `indu_t`는 과충전에서 결측, `indu_t.1`은 Test 전부 결측 | `indu_t_missing` 플래그, Train-only 중앙값 대체, `indu_t.1` 제외 | 결측 패턴을 활용하되 위험한 칼럼 배제 |
| 6단계에서 가스 센서 동시 0이 상당수 존재하며 실험별 분산 큼 | `gas_zero_count`, `gas_all_zero`를 단서로만 사용; 0을 안전/저단계로 단정하지 않음 | 결측/센서 무반응 상황에서도 학습 |
| `thermal_t=510` 값 일부 관측 | 값 그대로 보존 + `thermal_t_at_510` 플래그 | 센서 상한 도달/특이값을 표현 |
| 이미지 RGB 의사색상, 파일마다 크기가 다르고 색→절대온도 환산 불가 | ConvNeXt-Tiny, 비율 보존 패딩 후 224×224, 색상 증강 미사용; `thermal_t` 수치는 센서 가지로 전달 | 이미지의 공간 패턴이 센서와 상보적 정보 제공 |
| 5→6에서 이미지 변화량(MAE)이 항상 증가하지 않음 | 이미지 프레임 차분을 강제 신호로 쓰지 않고 현재 열화상 ConvNeXt 특징 사용 | 단일 프레임 의미 있는 정보 여부부터 검증 |

**유의:** 위 숫자와 관찰은 이전 EDA 및 대회 제공 설명을 바탕으로 정리했습니다. 각 특징이 실제로 도움이 되는지는 개별 실험으로 확인해야 하며, 서로 다른 시점·실험의 센서 중앙값 변화는 인과관계가 아닙니다.

## 2. 모델

```text
현재 열화상 PNG (RGB pseudo-color)
    └─ 비율 유지 + black letterbox → ImageNet normalization
    └─ ConvNeXt-Tiny (ImageNet pretrained, FROZEN) → 768D
    └─ Image projection MLP: 768 → 256 → 128

현재 + 같은 실험의 과거 센서 (172개 EDA 기반 특징)
    └─ Train-only median imputation + z-score scaling
    └─ Sensor MLP: 172 → 256 → 128

Gated Fusion (128-dimensional, sample-adaptive):
    gate = sigmoid(MLP([image_emb, sensor_emb]))
    fused = gate * image_emb + (1 - gate) * sensor_emb
    └─ 6-class classifier: 128 → 128 → 6

Auxiliary sensor-only head + image-only head
    loss = weighted_CE(fused) + 0.2 * weighted_CE(sensor) + 0.2 * weighted_CE(image)
```

- 본 모델은 **MLP와 이미지 투영층·게이트를 학습**합니다. ConvNeXt 백본 자체는 현재 버전에서 고정(Frozen)합니다. 성능 확인 후 마지막 ConvNeXt stage를 소규모 학습률로 fine-tuning하는 후속 실험을 권장합니다.
- 게이트는 이미지 가중치를 고정 0.5로 섞는 것이 아니라 **각 샘플, 각 차원마다 0~1**로 학습합니다. `mean_image_gate_weight`는 기여도나 인과적 영향으로 직접 해석하지 않습니다.
- sensor-only/image-only head F1은 보조 헤드 결과이지 **독립적으로 별도 학습한 센서-only / 이미지-only baseline과 동일한 결과가 아닙니다.**
- 미래 레이블이나 전체 실험 길이, 전체 실험 최대값을 사용하지 않으며 `label`을 특징에 포함하지 않습니다.
- Test를 여러 파일로 나누거나 순서를 섞어도 기존 행의 특징이 달라지지 않도록 설계했습니다.
- 실제 모델 학습 전후에 `check_submission.py`를 반드시 돌리세요.

## 3. GPU 서버에서 설치/실행

프로젝트 폴더에 `uni_dthon_convnext_gated_v1.zip`을 업로드한 후:

```bash
cd ~/jihyun/battery_multimodal_v2
source .venv/bin/activate
python -m zipfile -e uni_dthon_convnext_gated_v1.zip .
cd uni_dthon_convnext_gated_v1

# 환경 확인. 현재 서버의 CUDA torch를 무심코 CPU 버전으로 재설치하지 마세요.
python -c "import torch,torchvision,sklearn,pandas,PIL; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python tests/test_causality.py
```

사전학습 ConvNeXt-Tiny 가중치가 로컬에 있다면 다음과 같이 실행합니다. **기존 ResNet18 가중치는 ConvNeXt와 호환되지 않습니다.**

```bash
python train.py \
  --data-root /home/ubuntu/data/battery_data_light \
  --pretrained-weights /절대경로/convnext_tiny_imagenet1k_v1.pth \
  --cv-folds 5 --epochs 22 --patience 4 \
  --batch-size 256 --image-batch-size 64 \
  --output weights/model.pt
```

사전학습 가중치 파일이 없는 경우, **대회 규정상 다운로드가 허용되고 서버 인터넷이 가능한 경우에만** TorchVision 공식 가중치를 받게 할 수 있습니다:

```bash
python train.py \
  --data-root /home/ubuntu/data/battery_data_light \
  --download-pretrained \
  --cv-folds 5 --epochs 22 --patience 4 \
  --output weights/model.pt
```

`--random-backbone-smoke-only`는 개발용 임의 가중치일 뿐이며 **실제 모델로 제출하지 마세요.** `torch`와 `torchvision`은 CUDA 버전 호환성이 중요하므로 현재 `.venv`에 적절히 설치된 상태면 불필요한 설치를 피하세요.

실험 시간이 부족하면 3-Fold(`--cv-folds 3`)와 epochs 12로 **탐색만** 먼저 할 수 있지만, 최종 CV 지표와 직접 비교할 때 설정을 일치시키세요.

### 학습 결과

- `weights/model.pt`: 이미지 백본+센서 MLP+게이트+분류기+Train-only 정규화 통계까지 저장. **다시 학습할 필요 없음**
- `weights/validation_report.json`: CV, 전체 Validation, 단계별 F1, 혼동행렬, 유도 방식별 성능, Method B 10/15초 검증
- `weights/model.pt`는 대략 **108 MiB**로 GitHub의 일반 Git 100 MiB 제한을 초과합니다. **공식 제출 가이드에 맞춰 접근 가능한 Drive 링크를 README에 추가**하고, 검토자가 `weights/model.pt`로 받을 수 있도록 하세요. Test 데이터/원본 이미지/학습 캐시는 업로드 금지.

## 4. DAKER CSV 생성

학습을 완료했다면 **동일 디렉터리에서**:

```bash
python test.py \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --output submission_convnext_gated_v1.csv

head -5 submission_convnext_gated_v1.csv
wc -l submission_convnext_gated_v1.csv   # 헤더 포함 3,452줄이어야 함
```

유일하게 요구되는 CSV 열은 **`sample_id,label`**, 모든 라벨은 정수 1~6입니다.

## 5. 공식 자가 점검

```bash
python check_submission.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --final submission_convnext_gated_v1.csv
```

다음 사항이 모두 통과해야 최종 제출 후보가 됩니다: 재현성, 재실행, 행 섞기, 실험 분리, 미래 행 제거, **검은 이미지로 바꿨을 때 일부 예측 변화**. 마지막 항목은 이미지 사용을 모델 구조로만 보장할 수 없으므로 **실제 학습 결과로 확인해야 하며, 통과 여부를 미리 보장할 수 없습니다**. 미통과면 모델 개선이 필요합니다. 검사 통과를 위해 인위적으로 라벨을 바꾸는 구현은 하지 않았습니다.

## 6. 평가할 핵심 지표

- 5-Fold Train GroupKFold Macro F1 평균 및 편차
- 독립 Validation Macro F1 (기존 B 0.8123, B + Peaks 0.8151은 별도 팀 모델 비교 기준)
- `3 → 2`, `5 → 4`, `5 → 6`, `6 → 5` 오분류
- `tr_method` 1(가열), 2(과충전)별 F1
- Validation **실제 10/15초 샘플을 남기고 특징을 다시 계산한** Method-B Macro F1
- 이미지 헤드·센서 헤드 F1, 이미지 게이트 평균, 최종 6-stage F1

## 7. 제출 및 공개 주의

1. 원본 `battery_data_light`, 이미지, 실험별 샘플 예측, Train/Validation 라벨 포함 파일은 GitHub에 올리지 마세요.
2. 모델 가중치는 공식 가이드에 따라 별도 링크로 제공하되, 접근 권한과 다운로드 후 `test.py` 실행을 실제 확인하세요.
3. 이 실험 폴더에서 `python test.py --input ... --image_root ... --output ...`가 실행됩니다. **최종 공식 GitHub에서는 이 폴더의 파일이 제출 레포의 루트에 있어야** 운영진이 같은 명령어를 실행할 수 있습니다.
4. 이미지는 Test에서 반드시 `--image_root`로 직접 읽습니다. 테스트 특화 PCA/스케일러 fit, Test 전처리 캐시, 전체 실험 통계를 사용하지 않습니다.
5. 대회에서 허용되는 사전학습 가중치·인터넷 다운로드 범위는 최종 규정을 확인하세요.

## 8. 패키지 구성

```text
uni_dthon_convnext_gated_v1/
  model.py                  특징 생성 + ConvNeXt + Gated Fusion + 정규화
  train.py                  Train 5fold GroupKFold + 정식 학습 + Validation
  test.py                   주최측 지정 단일 추론 인터페이스
  check_submission.py       운영진 제공 자가 점검 스크립트
  requirements.txt          라이브러리 목록 (torch 버전 고정 안 함)
  tests/test_causality.py   시간 인과성/순서 불변성 테스트
  weights/                  모델 학습 후 출력 (기본 ZIP에는 가중치 없음)
  .gitignore                모델 가중치/원본 데이터 실수 공개 방지
```


## 실제 학습 결과 (2026-11-15)

| Metric | Result |
|---|---:|
| Train GroupKFold Macro F1 | 0.8501 |
| Validation Macro F1 | 0.8162 |
| Validation Accuracy | 0.8880 |
| Sensor Auxiliary Head F1 | 0.7997 |
| Image Auxiliary Head F1 | 0.6898 |
| Method-B 10s Macro F1 | 0.8169 |
| Method-B 15s Macro F1 | 0.8179 |
| Heating Method Macro F1 | 0.8349 |
| Overcharge Method Macro F1 | 0.7499 |

### Pretrained model checkpoint

The trained checkpoint is approximately 107.62 MiB and is not committed
to GitHub because it exceeds the standard Git file-size limit.

Download link: **추가 예정**

Place the downloaded checkpoint at `weights/model.pt` before inference.

### Known limitations

- Stage 2 performance is weaker for overcharge experiments.
- Stage 5 to Stage 3 errors were concentrated in one validation experiment.
- The official submission checker must be run against the final checkpoint
  and final CSV before submission.
