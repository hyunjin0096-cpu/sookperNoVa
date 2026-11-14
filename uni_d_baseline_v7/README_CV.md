# Uni-DTHON CV v1.1 — 실험 그룹 기반 검증 패키지

**대상:** 열화상 이미지 + 센서의 6단계 분류. 아래 파일은 **기존 `train.py`, `model.py`, `test.py`를 덮어쓰지 않습니다.** CSV 원본·이미지·정답·가중치를 포함하지 않습니다.

## 0. 구조

```text
uni_d_baseline_v7/
  cv/
    common.py             train+validation 통합, 실험 그룹 분리, 분할 후보 점수
    build_folds.py        StratifiedGroupKFold 5-Fold, seed 탐색, 고정
    cv_audit.py           실험/행 수, 클래스, 실험 조건 분포 감사
    inner_split.py        NN early stopping용 내부 실험 분리
  features/
    causal_sensor_features.py    초 단위 과거 이력 특징, 행 순서 불변
    gap_view.py                  5/10/15초 간격 모사
  experiments/
    sensor_cv_example.py  센서-only HGB 모델로 CV 파이프라인 작동 예시 (최종 제출 금지)
    evaluate_oof.py       Pooled 5초 OOF F1, Fold·클래스·실험조건별 F1
    evaluate_holdout.py   기존 78→19 홀드아웃 비교 (학습 누수 검사는 별도)
  checks/
    shortcut_baseline.py  이미지 크기-only 지름길 진단 (최종 제출 금지)
    selfcheck.py          행 셔플, 실험 분리, 미래 제거, 빈 이미지 검사
    make_gap_view.py      5/10/15초 간격 시뮬레이션 CSV
  tests/
    test_cv_pipeline.py   합성 데이터만 사용하는 기본 검증
  README_CV.md
```

**필요 의존성:** pandas, numpy, scikit-learn, Pillow. GPU 서버 공용 `~/venv`에 이미 있는지 확인하세요. 기존 서버 라이브러리를 무조건 업그레이드하지 마세요.

모든 명령은 **GPU 서버의 개인 프로젝트 루트**에서 실행합니다.

```bash
source ~/venv/bin/activate
cd ~/jihyun/uni_d_baseline_v7
export D="$HOME/data/battery_data_light"
python -m unittest discover -s tests -v
```

## 1. Fold 생성 — 반드시 한 번만

```bash
python -m cv.build_folds \
  --train-csv "$D/train/train.csv" \
  --validation-csv "$D/validation/validation.csv" \
  --out-dir artifacts/cv --n-splits 5 --seed-start 42 --seeds 30
```

생성 파일:
- `artifacts/cv/folds_exp.csv`: **experiment_id,fold** (97개 실험에 대해 0~4 배정)
- `artifacts/cv/seed_search.csv`: 시드 후보의 분포 감사 점수
- `artifacts/cv/split_config.json`: 선택된 seed, 감사 기준

이미 존재하는 Fold 파일을 덮어쓰지 않습니다. **모든 모델 팀원이 이 파일 그대로 공유하여 사용**하세요. 가장 균형 잡힌 split은 *클래스 분포/실험 조건/행 수*만 보고 결정합니다. 모델 성능을 보고 split을 고르면 안 됩니다.

> 기존 동료가 만든 `folds_exp.csv`가 있다면 **즉시 새로 생성하지 말고**, 그 파일의 실제 헤더/구조와 기존 실험 결과를 먼저 비교하세요. 학습된 모델들의 OOF fold가 서로 달라질 수 있습니다.

## 2. Fold 감사

```bash
python -m cv.cv_audit \
  --train-csv "$D/train/train.csv" \
  --validation-csv "$D/validation/validation.csv" \
  --folds artifacts/cv/folds_exp.csv \
  --out-dir artifacts/cv/audit
```

감사 결과의 `fold_summary.csv`, `fold_class_counts.csv`, `fold_tr_method_groups.csv`, `fold_bat_class_groups.csv`, `fold_bat_cc_groups.csv`, `fold_soc_status_groups.csv`를 확인하세요. 동일 실험 중복/클래스 누락이 없는지 확인할 수 있습니다. **CV 감사만으로는 시간 특징 누수 여부를 검사하지 못합니다.**

## 3. 모델 구현에 Fold 연결하기

```python
from cv.common import load_labeled, attach_folds
from cv.inner_split import make_inner_split

labeled = attach_folds(
    load_labeled(f"{D}/train/train.csv", f"{D}/validation/validation.csv"),
    "artifacts/cv/folds_exp.csv"
)

for outer_fold in range(5):
    outer_train = labeled[labeled.fold != outer_fold].reset_index(drop=True)
    outer_valid = labeled[labeled.fold == outer_fold].reset_index(drop=True)

    inner_tr_pos, inner_val_pos = make_inner_split(outer_train, seed=2026)
    # 1) Inner Train에서 모델/Scaler 학습
    # 2) Inner Validation으로 early stopping epoch만 결정
    # 3) 모델 가중치를 새로 초기화하고 Outer Train 전체로 정해진 epoch 수 학습
    # 4) Outer Validation은 딱 1회 성능 측정·OOF 예측 저장
    # 5) 각 OOF 행: sample_id,pred_label,fold[,prob_1..prob_6]
```

**주의:** 위 코드는 fold 연결 예시이며 PyTorch의 `fit()` 인터페이스는 사용자 모델마다 달라 빈칸을 남겼습니다. 실제 신경망 Trainer는 별도로 연결해야 합니다. Inner 전처리도 Inner Train만 `fit`, Outer 재학습의 전처리는 Outer Train만 `fit`합니다. validation fold로 epoch를 고르거나 fold 전체에서 scaler `fit`하면 누수/낙관 편향이 생길 수 있습니다.

GBDT처럼 early stopping을 사용하지 않는 모델은 해당 내부 분할을 생략할 수 있습니다.

## 4. 작동 예시 — 센서-only (최종 제출용 아님)

처음에는 Fold 0, 1만 빠르게 검증:

```bash
python -m experiments.sensor_cv_example \
  --train-csv "$D/train/train.csv" \
  --validation-csv "$D/validation/validation.csv" \
  --folds artifacts/cv/folds_exp.csv \
  --fold-ids 0 1 --gap-check \
  --out artifacts/cv/sensor_screen_oof.csv
```

스크리닝 결과는 나머지 Fold가 없으므로 **모델 최종 비교에 사용하면 안 됩니다.**

유망 모델은 전체 5-Fold에서 테스트:

```bash
python -m experiments.sensor_cv_example \
  --train-csv "$D/train/train.csv" \
  --validation-csv "$D/validation/validation.csv" \
  --folds artifacts/cv/folds_exp.csv \
  --fold-ids 0 1 2 3 4 --gap-check \
  --out artifacts/cv/sensor_full_oof.csv
```

이 코드는 `HistGradientBoostingClassifier`를 CPU에서 돌리는 CV 배선 점검 예시입니다. **대회 최종 모델은 실제 이미지와 센서를 모두 이용해야 하며, 센서-only 결과를 제출하면 규정을 충족하지 못합니다.**

## 5. OOF 성능 집계

```bash
python -m experiments.evaluate_oof \
  --train-csv "$D/train/train.csv" \
  --validation-csv "$D/validation/validation.csv" \
  --folds artifacts/cv/folds_exp.csv \
  --oof artifacts/cv/sensor_full_oof.csv \
  --out-dir artifacts/cv/sensor_eval
```

- `pooled_oof_macro_f1`: **주 지표** (원본 5초 행 전체 43,722개 예측에 대해 Macro F1)
- `fold_f1_mean`, `fold_f1_std_sample`: Fold 안정성. 평균과 pooled 값은 달라질 수 있음
- `class_metrics.csv`, `subgroup_metrics.csv`, `confusion_matrix.csv`

스크리닝 0·1만 평가할 때만 `--allow-partial` 추가. partial은 완전한 OOF 지표 아님.

**동일 조건 78→19 Holdout**은 기존 baseline 0.6674와 비교 가능하나, 해당 모델이 원본 Train 78만으로 학습됐는지 별도 확인해야 합니다.

```bash
python -m experiments.evaluate_holdout \
  --validation-csv "$D/validation/validation.csv" \
  --predictions artifacts/holdout_preds.csv
```

`holdout_preds.csv`의 열은 `sample_id,pred_label`입니다.

## 6. Test 간격 모사 및 이미지 지름길

`--gap-check`: **동일 평가 행**에 대해 전체 과거 이력과 일부 관측이 제거된 과거 이력으로 재계산한 센서 예측을 비교합니다. 비율은 5·10·15초 **기본 1/3씩**이며 Test CSV를 읽어 확률을 조정하지 않습니다. 이 보조 지표를 원본 5초 OOF와 직접 비교하지 마세요.

이미지 너비/높이/비율만을 특징으로 하는 지름길 진단:

```bash
python -m checks.shortcut_baseline \
  --data-root "$D" \
  --folds artifacts/cv/folds_exp.csv \
  --out-dir artifacts/shortcuts
```

`artifacts/shortcuts/size_only_oof.csv`를 위 5번 OOF 평가 명령에 넣어 비교하세요. **이 진단 모델은 절대로 최종 제출용이 아닙니다.**

## 7. 운영진 재현성 검사 4종

최종 **이미지+센서 모델**을 저장한 뒤, 기존 `test.py`가 다음 CLI를 지원할 때:

```bash
python -m checks.selfcheck \
  --input "$D/test/test.csv" \
  --image-root "$D/test" \
  --test-script test.py --experiments 2
```

검사 항목: (1) 행 셔플 예측 동일 (2) 같은 실험만 넣어도 예측 동일 (3) 이후 행 제거해도 기존 시점 예측 동일 (4) 이미지를 검게 바꿨을 때 변화가 있는지 확인. 수백~수천 장을 임시 생성할 수 있으니 디스크/시간 확인. **과거·현재 정보로만 예측해야 합니다.**

`test.py`에 추가 CLI 옵션이 필요하면 `--extra-args-json '["--model", "model.joblib"]'`로 지정하세요. 옵션 이름은 기존 `test.py` 구현에 맞춰야 합니다.

주의: 검은 이미지로 바꾼 뒤 모든 argmax 라벨이 그대로라면 이미지가 실제로 영향을 주지 않았거나, 확률만 달라졌을 수 있습니다. 확률 차이도 확인하면 더 정확합니다. 자체 검사는 운영진의 공식 재현성 검증을 대체하지 않습니다.

## 8. 최종 모델 선택 및 제출

1. 동료들과 동일 frozen fold 5개로 각 모델의 OOF 비교 (여러 실험 과적합 경계)
2. 센서-only, 이미지-only는 진단용, 제출은 **이미지와 센서 모두 사용**
3. 최종 구조·epoch·전처리·앙상블 결정 후 **97개 실험 모두로 재학습**
4. 모델 가중치와 전처리 객체 저장. 추론 `test.py`는 저장 객체를 로드해서 **재학습 없이** 출력
5. `checks.selfcheck` 통과, sample_submission의 sample_id/행 순서 일치, 정수 1..6 확인
6. GitHub에는 코드와 **최종 가중치(OT 규정)** 포함, 대회 데이터와 파생 파일은 제외

기존 `.gitignore`에서 `*.joblib`을 무조건 제외했다면 최종 제출 시 필요한 가중치에 대해 `git add -f <모델파일>`처럼 명시적인 예외 처리가 필요할 수 있습니다. **모델 크기·저장소 공개 규정도 확인하세요.**

## 9. 규정/해석 주의

- `Train/Validation` 라벨은 대회에서 학습에 사용 가능; Test는 추론 전용입니다.
- `test.csv`의 실제 5/10/15 비율을 학습 증강에 맞추지 않습니다.
- Train+Validation 전체에서 신경망 scaler를 미리 학습하면 CV Fold의 평가 분포가 유출됩니다. **Fold별 학습 데이터에서만 fit** 합니다.
- StratifiedGroupKFold의 shuffle은 모든 그룹을 자유롭게 섞는 방식이 아닙니다. 균형 감사 결과를 확인하세요.
- Pooled OOF로 앙상블 가중치/클래스 offset을 최적화하면 그 OOF 점수는 낙관적일 수 있습니다.
- `artifacts/`, 원본 이미지, 수치가 포함된 분석 출력, `folds_exp.csv`는 팀 서버 내부에 보관합니다.
