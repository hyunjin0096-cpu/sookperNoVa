# Uni-D 배터리 열폭주 v4: 기존 B 190개 + Frozen DINOv2

## 준비
- `model_v3_noix.joblib`는 기존 B 모델(190개 특징)의 저장본. **재학습 없이 성능 비교 모델로 로드되는 것이 아니라**, 이 파일의 정확한 190개 열 이름·피처 생성 규격·class_weight를 고정하여 각 모델을 동일한 설정으로 새로 학습합니다.
- `model.py`는 업로드한 `model(1).py`를 이름만 바꾼 파일입니다. 원본 피처 생성 로직은 변경하지 않았습니다.
- 이미지 파일과 train/train.csv 및 validation/validation.csv가 필요합니다. 업로드에는 원본 이미지 데이터가 없으므로 점수는 여기서 실측할 수 없습니다.
- 첫 실행 시 `torch.hub`가 Facebook Research DINOv2 코드와 **사전학습 가중치**를 받아옵니다. 인터넷이 필요하며 모델 라이선스 및 대회 외부 사전학습 모델 허용 여부를 확인하세요.
- 가급적 CUDA GPU 권장. GPU 없는 경우 batch_size를 4~8로 줄이세요.

## 설치
```bash
pip install -r requirements_v4.txt
```

## 학습·검증 (프로젝트 폴더에서)
```bash
python train_v4.py \
  --data_dir ./data/battery_data_light \
  --baseline_model ./model_v3_noix.joblib \
  --cache_dir ./cache_v4 \
  --output_dir ./outputs_v4 \
  --batch_size 32
```
이 명령은 baseline, PCA64, PCA128, full384를 동일한 GroupKFold(실험 ID 단위 5-fold)로 검증하고, 별도 validation 원본 간격 및 15초 간격 점수와 클래스별 F1, 혼동행렬을 `outputs_v4/comparison_v4.json`에 저장합니다. `oof_v4_*.csv`와 `model_v4_*.joblib`도 저장합니다.

PCA는 각 fold의 Train 그룹의 **원본(5초) 샘플 임베딩으로만** fit합니다. 해당 fold의 Validation 실험은 사용하지 않습니다. 최종 모델 PCA 역시 Train만으로 fit합니다. ViT 가중치는 모든 단계에서 동결합니다. 각 모델의 시계열 간격 증강은 기존 B 모델과 같은 10·15초 subsampling을 사용합니다. 15초 검증에서는 시계열 특징을 다시 계산합니다.

`comparison_v4.json`의 `validation_5s.per_class_f1["4"]` 및 `["5"]`와 혼동행렬 `[4행,5열]`, `[5행,4열]`을 확인하세요. 모델 선택은 우선 Train GroupKFold 점수로 하고, Validation 점수는 참고로 보고하세요. 단조성 보정은 비활성화했습니다.

## 추론·제출
```bash
python test_v4.py \
  --input ./data/battery_data_light/test/test.csv \
  --image_root ./data/battery_data_light/test \
  --model ./outputs_v4/model_v4_pca64.joblib \
  --output ./submission_v4.csv \
  --sample_submission ./data/battery_data_light/sample_submission.csv \
  --batch_size 32
```
위의 모델 경로는 **예시**이므로 GroupKFold 비교 후 선택된 것으로 변경합니다. 테스트 입력 CSV 안에 있는 행들의 과거 관측 정보만으로 기존 190개 시계열 특징을 계산합니다. ViT 임베딩은 해당 이미지 하나로부터만 추출합니다. 추론 환경에서도 최초 가중치 다운로드 가능해야 하므로 대회 제출 서버의 네트워크/모델 캐시 조건을 확인하세요.

### 주의
- DINOv2 입력은 224×224 RGB, ImageNet 정규화; `x_norm_clstoken` 384차원 사용. 이미지를 학습 데이터로 **fine-tuning하지 않음**.
- PCA는 라벨 없이 fit하더라도 CV fold 바깥의 Validation 데이터를 보지 않아야 누출 없는 평가가 됩니다.
- 원본 baseline 성능과 새로 학습한 baseline은 버전/라이브러리/데이터에 따라 미세하게 다를 수 있습니다.
- v4의 15초 검증은 각 실험에서 3개 행 중 첫 행을 남기는 시뮬레이션입니다. 실제 제출 분포를 완전히 재현하는 것은 아닙니다.
