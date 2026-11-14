# Uni-DTHON v1 — 50:50 Sensor HGB + Fusion NN 최종 제출 연결

**목표:** 기존 CV 결과 (HGB 0.859112, Fusion 0.822820, 50:50 0.858031)를 토대로, **Test 라벨을 사용하지 않고** 전체 97개 라벨 실험에서 **센서 모델만 최종 재학습**, 이미 존재하는 **전체 데이터 학습 Fusion NN 체크포인트**를 재사용, 최종 `test.py`로 제출 파일 생성.

**확인된 사실:** Fusion NN 단독의 DAKER Public 점수 0.7533384075. 앙상블 Public 점수는 아직 측정하지 않았으며 더 높다는 보장은 없습니다. CV에서는 50:50이 센서 단독보다 약 0.00108 낮지만, 센서 단독 제출은 이미지 이용 규정을 충족하지 못합니다. 정상 vs 빈 이미지 OOF에서 50:50 앙상블의 Macro F1은 0.85803 -> 0.77016, 예측 변경률 14.38% (단, 이로부터 이미지의 물리적 의미가 증명되는 것은 아닙니다).

## 새로 추가되는 파일

- `train_sensor_ensemble.py`: 기존 `sensor_cv_example.py` **동일 피처 및 HGB 하이퍼파라미터**로 Train+Validation 전체 학습, `weights/ensemble_v1/sensor_model.joblib` 저장. 새 모델 학습은 이 단계 **한 번**만 진행.
- `scripts/prepare_ensemble.py`: 기존 `weights/fusion_v1` 최종 Fusion 체크포인트를 `weights/ensemble_v1/fusion/`로 복사. 학습 없음.
- `test_ensemble.py`: 지정된 입력 CSV와 이미지 폴더, 로컬 `weights/ensemble_v1`만 사용. HGB 확률과 이미지 Fusion NN 확률을 50:50으로 평균, `sample_id,label` 저장. 학습/fit 없음.
- `tests/test_ensemble_submission.py`: 앙상블 연결 단위 테스트.

기존 `model.py`, `train.py`, `test.py`, OOF 및 Fold를 **자동 수정하지 않습니다**.

## 1. Windows CMD 업로드

```cmd
scp "%USERPROFILE%\Downloads\UniD_Ensemble_Submission_v1_0.zip" ubuntu@118.216.110.86:/home/ubuntu/jihyun/
```

Ubuntu 서버 VS Code 터미널:

```bash
source ~/venv/bin/activate
cd ~/jihyun/uni_d_baseline_v7
ls -lh ~/jihyun/UniD_Ensemble_Submission_v1_0.zip
python3 -m zipfile -e ~/jihyun/UniD_Ensemble_Submission_v1_0.zip .
python -m unittest discover -s tests -v
```

## 2. 기존 전체 Fusion 체크포인트 확인

```bash
ls -lh weights/fusion_v1/model.pt weights/fusion_v1/sensor_preprocess.json weights/fusion_v1/config.json
```

이 파일들이 없다면 **별도 경로에 저장했는지 먼저 확인**하고, `--source`로 정확한 최종 체크포인트 경로를 지정하세요. CV용 `artifacts/fusion_cv/fold_0/` 등은 **최종 97개 학습 모델과 다릅니다**.

## 3. 최종 GBDT 센서 모델 학습 및 저장 (Test 사용 금지)

```bash
D=~/data/battery_data_light
python -u train_sensor_ensemble.py \
  --data-dir "$D" \
  --out-dir weights/ensemble_v1 \
  2>&1 | tee sensor_final.log
```

기존 실험과 같은 센서 피처 + `HistGradientBoostingClassifier(max_iter=75, max_leaf_nodes=31, learning_rate=0.07, class_weight='balanced', l2_regularization=1.0, early_stopping=False, random_state=42)`. 센서-only 모델은 이대로 제출 불가.

## 4. Fusion 최종 체크포인트 복사 (학습 안 함)

```bash
python -m scripts.prepare_ensemble \
  --source weights/fusion_v1 \
  --dest weights/ensemble_v1
```

필요 파일 확인:

```bash
ls -lh weights/ensemble_v1/sensor_model.joblib
ls -lh weights/ensemble_v1/fusion/model.pt
ls -lh weights/ensemble_v1/fusion/sensor_preprocess.json
ls -lh weights/ensemble_v1/ensemble_config.json
```

`--fp16-storage` 옵션을 추가하면 저장된 NN 실수 가중치를 절반 정밀도로 변환해 GitHub 25MB 웹 업로드 제한을 통과하기 쉽게 만들 수 있습니다. **다만 예측이 소폭 변할 수 있으므로 반드시 새로 `test.py`를 돌려 생성한 CSV를 제출하고, 이미지 기여/공식 재현성 테스트도 다시 진행**해야 합니다. 초기에는 FP32를 보존하는 기본 방식이 안전합니다. 25MB 초과 파일은 운영진의 Drive 링크/제출 지침을 따라야 하며, `test.py`가 실행되는 위치에서 로컬 가중치 파일을 실제로 읽을 수 있어야 합니다.

## 5. 첫 앙상블 Test CSV 생성

먼저 `test_ensemble.py`로 동작을 확인합니다.

```bash
python test_ensemble.py \
  --input "$D/test/test.csv" \
  --image_root "$D/test" \
  --output submission_ensemble_v1_draft.csv
```

행 수 및 순서 검증:

```bash
python - <<'PY'
import pandas as pd
D='/home/ubuntu/data/battery_data_light'
test=pd.read_csv(D+'/test/test.csv')
sub=pd.read_csv('submission_ensemble_v1_draft.csv')
assert len(test)==len(sub)==3451
assert sub.columns.tolist()==['sample_id','label']
assert sub.sample_id.tolist()==test.sample_id.tolist()
assert sub.label.isin(range(1,7)).all()
print('SUBMISSION FORMAT OK')
print(sub.label.value_counts().sort_index())
PY
```

## 6. 공식 test.py로 안전하게 전환

앞서 DAKER에 낸 Fusion CSV와 해당 `test.py`를 잃지 않도록 보존합니다. 운영진은 **`test.py`만 실행**합니다.

```bash
cp test.py test_baseline_backup.py
cp test_ensemble.py test.py

# 이 CSV가 실제 DAKER에 제출할 후보가 됩니다.
python test.py \
  --input "$D/test/test.csv" \
  --image_root "$D/test" \
  --output submission_ensemble_v1.csv

cmp submission_ensemble_v1_draft.csv submission_ensemble_v1.csv && echo 'REPRODUCIBLE: IDENTICAL CSV'
```

후보가 채택되면 운영진 공식 `check_submission.py`를 **공식 명령으로 실행**하고, 행 섞기/실험 나누기/미래 행 제거, 동일 입력 반복, 이미지 바꾸기 반응을 확인합니다. 기존 보조 검사도 가능합니다:

```bash
python -m checks.selfcheck \
  --input "$D/test/test.csv" \
  --image-root "$D/test" \
  --test-script test.py \
  --out-dir artifacts/ensemble_selfcheck
```

**공식 검사**(운영진 제공 파일이 프로젝트에 있다면):

```bash
python check_submission.py --data_dir "$D" --final submission_ensemble_v1.csv
```

**다른 CSV를 최종 선택하면** 해당 CSV를 만든 모델/가중치와 같은 `test.py`로 되돌려야 합니다. 최종 파일 이름만 바꾸거나 서로 다른 모델의 출력을 혼용하면 안 됩니다.

## 7. GitHub 및 최종 제출 참고

- **필수:** `model.py`, `train.py`, `test.py`, `README.md`, `requirements.txt`, 모든 의존 모듈 `features/`, `fusion_model.py`, `test_fusion.py`, `weights/` 및 필요한 연결 코드. `train.py`가 최종 모델을 재현하도록 `train_sensor_ensemble.py` + `train_fusion.py` 학습 순서를 README에 명시하거나 `train.py`에 통합하세요. 현재 압축 파일은 `test.py`와 `train.py`를 자동 변경하지 않습니다.
- `requirements.txt`에 `numpy`, `pandas`, `scikit-learn`, `joblib`, `Pillow`, `torchvision`, `torch` 등을 명시하고 **torch 버전은 적지 마세요**(운영진 지침). 실제 환경 버전은 README에 기록하세요.
- GitHub에 제공 데이터, 제공 이미지, OOF/CSV 행·실험 ID, 실제 대회 학습 파일을 올리지 마세요. 코드·검증 설정·최종 가중치만 제출하세요. `.gitignore`를 확인하세요.
- 25MB가 넘는 체크포인트는 운영진 지침에 따라 Drive 링크를 README에 제공하되, 실제 코드 실행 시 가중치가 올바른 위치에 있어야 합니다. Drive 링크가 `test.py`의 파일 자동 다운로드를 대체하지 않습니다.
- DAKER Public 점수만 보고 앙상블 가중치를 과도하게 최적화하지 마세요.

## 주의사항

1. 50:50 확률 OOF 0.858031은 **OOF** 점수이고, 이번 전체 97개 학습 Test 앙상블의 DAKER 점수는 **아직 미확인**입니다.
2. 운영진은 Test를 실험별로 나누거나 미래 행을 제거해 실행하므로, 입력 CSV에 없는 관측값/다른 CSV/데이터 전체 통계를 참조하면 안 됩니다.
3. 정상/빈 이미지 기여도 검사는 **최종 가중치**에서 재실시할 필요가 있습니다. OOF 검사 결과를 최종 모델에 그대로 일반화할 수 없습니다.
4. 딥러닝 모델 가중치가 FP32 약 44MB이면 웹 GitHub 업로드 25MB 제한을 넘을 수 있습니다. 공식 안내에 따라 큰 가중치는 Drive 링크를 추가하세요. CLI push 100MB 미만 여부와 별개로 운영진 제출 규칙을 확인하세요.

## 8. 운영진용 train.py / model.py 정리 (앙상블을 최종 선택한 경우)

기본 `train.py`와 `model.py`는 이전 베이스라인 코드일 수 있으므로, 최종 앙상블을 선택한 뒤에만 백업하고 바꿉니다.

```bash
cp train.py train_baseline_backup.py
cp model.py model_baseline_backup.py
cp train_ensemble.py train.py
cp model_ensemble.py model.py
```

`train.py`를 실행하면 기본적으로 전체 라벨 데이터를 이용해 센서 모델과 Fusion NN을 **모두 재학습**합니다. 이번 대회에서 이미 학습해 둔 Fusion 체크포인트를 재사용하는 재현 명령은 다음과 같습니다.

```bash
python train.py --data-dir "$D" --out-dir weights/ensemble_v2 \
  --reuse-fusion-final weights/fusion_v1
```

새 폴더를 사용해야 기존 `weights/ensemble_v1` 체크포인트를 덮어쓰지 않습니다. `requirements_ensemble.txt`의 내용을 최종 `requirements.txt`에 병합하세요. `torch`의 버전은 운영진 지침에 따라 고정하지 않습니다.
