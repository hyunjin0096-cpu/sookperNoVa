# Uni-DTHON Fusion v1.1 — CV + ResNet18 + Sensor MLP

**추가 파일만 제공**합니다. 기존 `model.py`, `train.py`, `test.py`, `folds_exp.csv`, `model.joblib`은 절대 덮어쓰지 않습니다. 앞서 설치한 CV 패키지와 Preprocessing 패키지가 필요합니다.

## 구성

- `fusion_model.py`: 이미지 ResNet18(512→128) + Sensor MLP(149→64) + 결합 분류기(192→6).
- `experiments/train_fusion.py`: 고정 5-Fold CV 또는 전체 97개 실험 학습. Fold 검증에서 Epoch를 선택하지 않음.
- `train_fusion.py`: 최종 `train.py`로 복사할 학습 진입점.
- `test_fusion.py`: **모델을 절대 fit하지 않는** 제출 형식 추론(`--input --image_root --output`), `weights/`의 저장 모델만 로드.
- `scripts/fusion_smoke.py`: 가상 센서·이미지로 순전파, 역전파, 가중치 저장·복원 검증.
- `scripts/merge_fusion_oof.py`: 모든 Fold의 OOF 파일 검증 후 병합.
- `tests/test_fusion_model.py`: 아키텍처 단위 테스트.

## 0. 서버 설치

Windows CMD에서:

```cmd
scp "%USERPROFILE%\Downloads\UniD_Fusion_v1_1.zip" ubuntu@118.216.110.86:/home/ubuntu/jihyun/
```

VS Code **Ubuntu 서버** 터미널에서:

```bash
source ~/venv/bin/activate
cd ~/jihyun/uni_d_baseline_v7
ls -lh ~/jihyun/UniD_Fusion_v1_1.zip
python3 -m zipfile -e ~/jihyun/UniD_Fusion_v1_1.zip .
```

현재 서버의 torch/torchvision 호환성과 GPU 지원을 확인하세요. **GPU 사용 여부 출력이 중요합니다.**

```bash
python - <<'PY'
import torch, torchvision
print('torch', torch.__version__, 'torchvision', torchvision.__version__)
print('CUDA',torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU',torch.cuda.get_device_name(0))
    print('test', (torch.ones(1, device='cuda') + 1).item())
PY
```

CUDA가 False라면 **절대 15 Epoch 전체 학습부터 시작하지 마세요.** 제공된 대회 서버의 PyTorch/CUDA 환경을 먼저 점검해야 합니다.

## 1. 무해한 모의 데이터 검사 (실제 데이터 학습 X)

```bash
python -m unittest discover -s tests -v
python -m scripts.fusion_smoke
```

예상: `OK`, `FUSION SYNTHETIC SMOKE OK`.

## 2. GPU 학습 배선 확인 — Fold 0, 2 배치, 1 Epoch

```bash
D=~/data/battery_data_light
sha256sum -c artifacts/cv/folds_exp.sha256
python -m experiments.train_fusion \
  --data-dir "$D" --mode cv --fold-ids 0 \
  --epochs 1 --batch-size 32 --num-workers 2 --smoke
```

예상: `FUSION TRAIN SMOKE OK — no model files saved`.

**SMOKE 점수는 학습 샘플이 극히 적으므로 CV 모델 성능으로 사용하지 않습니다.**

## 3. Fold 0 학습 — 첫 실제 평가

```bash
python -u -m experiments.train_fusion \
  --data-dir "$D" --mode cv --fold-ids 0 \
  --epochs 15 --batch-size 64 --num-workers 4 \
  --out-dir artifacts/fusion_cv \
  2>&1 | tee fusion_fold0.log
```

결과:

```text
artifacts/fusion_cv/fold_0/model.pt
artifacts/fusion_cv/fold_0/sensor_preprocess.json
artifacts/fusion_cv/fold_0/config.json
artifacts/fusion_cv/fold_0/oof.csv
```

**ResNet18은 기본적으로 랜덤 초기화**됩니다. 대회 규정에 따라 공개 사전학습 모델 사용이 허용된다면 `--pretrained-backbone`을 추가할 수 있으나, 최초 실행 시 torchvision이 공개 ImageNet 가중치를 다운로드할 수 있습니다. 사전학습 가중치를 사용한 경우 반드시 README에 출처와 버전을 명시하세요. 제출 `test_fusion.py`는 **외부 다운로드를 절대 하지 않으며** `weights/model.pt`만 로드합니다. 동일한 CV 비교에서 초기화/사전학습 여부가 다르면 별도 실험으로 기록하세요.

초기 베이스라인에 비해 CV 비교가 공정하려면, 센서 GBDT도 **동일한 고정 5-Fold**로 OOF를 생성해야 합니다. 기존 Train→Validation 점수 0.6674는 통합 CV 점수와 직접 비교할 수 없습니다.

## 4. 5-Fold 전체 학습 (각 Fold 결과를 별도로 보존)

```bash
python -u -m experiments.train_fusion \
  --data-dir "$D" --mode cv --fold-ids 1 2 3 4 \
  --epochs 15 --batch-size 64 --num-workers 4 \
  --out-dir artifacts/fusion_cv \
  2>&1 | tee fusion_folds1to4.log
```

이미 존재하는 모델을 실수로 덮어쓰지 않게 기본적으로 파일 존재 여부를 검사합니다. **Fold 0은 다시 학습하지 않으므로** 동일한 설정으로 학습한 Fold 1~4와 합칠 수 있습니다. 서로 다른 설정의 OOF를 같은 실험으로 병합하지 마세요.

```bash
python -m scripts.merge_fusion_oof \
  --data-dir "$D" --model-dir artifacts/fusion_cv \
  --output artifacts/fusion_cv/fusion_oof.csv

python -m experiments.evaluate_oof \
  --train-csv "$D/train/train.csv" \
  --validation-csv "$D/validation/validation.csv" \
  --folds artifacts/cv/folds_exp.csv \
  --oof artifacts/fusion_cv/fusion_oof.csv \
  --out-dir artifacts/fusion_cv/evaluation
```

## 5. 최종 모델 — 97개 라벨 실험으로 재학습

**모델/학습 설정을 선택한 뒤** 최종 모델을 학습합니다. CV의 outer fold가 아니라 전체 97개에서 학습하므로 Fold 파일은 추론에 필요 없습니다.

```bash
python -u -m experiments.train_fusion \
  --data-dir "$D" --mode final \
  --epochs 15 --batch-size 64 --num-workers 4 \
  --out-dir weights \
  2>&1 | tee fusion_final.log
```

결과: `weights/model.pt`, `weights/sensor_preprocess.json`, `weights/config.json`.

ResNet18 전체 FP32 가중치는 **약 44MB**로 25MB를 넘을 수 있습니다. **운영진이 지정한 Drive 링크 방식**으로 접근 가능하게 하고 README에 정확한 받는 방법을 적어야 합니다. 저장 가중치는 최종 `test.py` 실행 전에 `weights/model.pt` 경로에 있어야 합니다. 최종 GitHub에는 제공 데이터 원본, 이미지, 실험별 Fold 배정 CSV를 올리지 마세요.

## 6. 제출용 `test_fusion.py` 검증

```bash
python test_fusion.py \
  --input "$D/test/test.csv" \
  --image_root "$D/test" \
  --output submission_fusion.csv
```

기본은 **배치 크기 1**, 저장 가중치 로드, scaler load-only, `model.eval()` + `torch.inference_mode()`. 입력 행 순서를 유지하며 정확히 `sample_id,label`만 출력합니다.

이어서 먼저 자체 점검:

```bash
python -m checks.selfcheck \
  --input "$D/test/test.csv" \
  --image-root "$D/test" \
  --test-script test_fusion.py \
  --experiments 2
```

여기서 앞의 3가지 불변성은 반드시 True여야 합니다. 빈 이미지 결과는 **변화 여부뿐 아니라 클래스별 결과와 확률 수준에서도 분석**하세요. 빈 이미지에서 라벨이 안 바뀐다고 자동으로 규정 위반 확정은 아니지만, 이미지가 실제로 기여하는지 설명할 수 있어야 합니다.

**최종 Fusion을 선택했을 때만** 원본 `train.py`, `test.py`, `requirements.txt`를 별도 이름으로 백업하고 교체합니다:

```bash
cp test.py test_baseline_backup.py
cp train.py train_baseline_backup.py
cp requirements.txt requirements_baseline_backup.txt
cp test_fusion.py test.py
cp train_fusion.py train.py
cp requirements_fusion.txt requirements.txt
python test.py --input "$D/test/test.csv" --image_root "$D/test" --output submission_fusion_final.csv
```

**최종 리더보드에 선택한 CSV는 이 `test.py`로 만든 파일과 동일해야 합니다.** 이후 운영진이 제공한 `check_submission.py`를 반드시 실행하세요. 모델을 바꾸거나 가중치를 재학습하면 CSV를 다시 생성하여 일치시켜야 합니다.

## 안전 원칙

- CV 폴드 맵을 재생성하지 않음. `sha256sum -c ...`로 검증.
- 오직 현재·과거 같은 실험의 관측값만 이용. 입력에 없는 과거 행은 조회하지 않음.
- CV 전처리 통계는 해당 외부 Train 그룹에서만 fit.
- Epoch는 15로 **미리 고정**, 외부 Validation 성능으로 최적 Epoch를 선택하거나 조기 종료하지 않음.
- 대회 원본 이미지의 의사색상은 색조 변형 금지; 기본 160px Pad. Resize/stretch는 별도 실험.
- 기본 CE, Weighted CE는 `--weighted-ce` 별도 실험.
- RTX 5080 환경의 torch/torchvision/CUDA 호환성을 확인하고 GPU 사용 여부를 출력.
- Final 모델이 센서를 무시하는지 / 이미지를 무시하는지 별도 확인해야 함. 센서-only는 **대회 최종 제출 불가**.
