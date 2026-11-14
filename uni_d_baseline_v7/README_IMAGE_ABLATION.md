# Uni-DTHON 이미지 기여도 검사 (v1.0)

**목적:** 저장한 Fusion 모델을 다시 학습하지 않고, 동일 샘플의 열화상 이미지가 정상일 때와 검은색으로 교체됐을 때를 비교한다. 센서값, 현재·과거 피처, 모델 가중치는 동일하게 유지한다. 모든 모델은 각 Fold 학습 데이터로 학습한 가중치를 자기 검증 Fold에만 적용한다.

기존 `train.py`, `model.py`, `test.py`, `fusion_model.py`, `folds_exp.csv`, 모델 가중치를 변경하지 않는다. 새 파일 `scripts/image_ablation.py`만 추가한다.

## 1. 업로드

Windows CMD:

```cmd
scp "%USERPROFILE%\Downloads\UniD_Image_Ablation_v1_0.zip" ubuntu@118.216.110.86:/home/ubuntu/jihyun/
```

GPU 서버 Ubuntu 터미널:

```bash
source ~/venv/bin/activate
cd ~/jihyun/uni_d_baseline_v7
ls -lh ~/jihyun/UniD_Image_Ablation_v1_0.zip
python3 -m zipfile -e ~/jihyun/UniD_Image_Ablation_v1_0.zip .
ls -l scripts/image_ablation.py
```

## 2. 해시 검증 및 실행

```bash
D=~/data/battery_data_light
sha256sum -c artifacts/cv/folds_exp.sha256
python -u -m scripts.image_ablation \
  --data-dir "$D" \
  --folds artifacts/cv/folds_exp.csv \
  --model-dir artifacts/fusion_cv \
  --fusion-oof artifacts/fusion_cv/fusion_oof.csv \
  --sensor-oof artifacts/cv/sensor_example_oof.csv \
  --out-dir artifacts/image_ablation \
  2>&1 | tee image_ablation.log
```

원본 학습 OOF 검증과 최대한 일치하게 하기 위해 기본 배치 크기를 64, CUDA AMP를 ON으로 설정했다. `recompute_vs_saved_oof.label_mismatch_rate`가 0% 부근이어야 하며 0.5%가 넘으면 스크립트 버전, 배치 크기, AMP 차이를 먼저 검사해야 한다. 기본 설정은 모든 5-Fold, 모든 43,722행 평가다.

## 3. 출력

- `artifacts/image_ablation/summary.json`: 정상/빈 이미지 Fusion F1, 변경 라벨 비율, 확률 변화량, 50:50 앙상블 비교
- `artifacts/image_ablation/fold_metrics.csv`: Fold별 정상/빈 이미지 점수와 변화
- `artifacts/image_ablation/class_metrics.csv`: 정상/빈 이미지 클래스별 지표
- `artifacts/image_ablation/image_ablation_oof.csv`: 샘플별 확률 및 진단값. **대회 데이터 파생 산출물로서 Public GitHub 업로드 금지**

## 4. 판정

- 이미지 교체 후 예측 확률이 바뀌는지: **필요한 진단**. 확률 변화만으로 실제 열패턴 활용이 입증되지는 않는다.
- 최종 클래스가 바뀌는 샘플이 있는지: **운영진의 빈 이미지 검사와 직접 관련**. 전혀 바뀌지 않으면 특히 주의.
- 정상 이미지가 빈 이미지보다 F1이 높은지: 일반화 가능한 이미지 정보가 유효할 가능성을 확인. 반대로 더 낮으면 이미지 브랜치가 방해가 될 수 있다.
- 50:50 앙상블을 제출 후보로 쓴다면 앙상블 자체가 빈 이미지에 반응해야 한다.

이 진단은 CV 평가용이므로 최종 `test.py`의 실제 Test 분리·행 섞기·미래 행 제거 규정 검사를 대체하지 못한다. 최종 가중치와 제출 CSV는 별도 준비가 필요하다.
