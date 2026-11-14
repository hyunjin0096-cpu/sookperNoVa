# Uni-D 배터리 열폭주 6단계 분류 — v8 (LightGBM + 의사색 이미지 피처 + 다중 윈도우 시계열)

> `<>` 부분은 제출 전에 팀 내용으로 채우십시오.

## 팀 정보
- 팀명: <팀명>
- 최종 선택 제출 파일: <리더보드에서 선택한 파일명>, Private 복원 대상

## 개발 환경
- OS: <예: Ubuntu 22.04>  /  Python: <예: 3.10>  /  GPU: 미사용 (LightGBM CPU)
- 라이브러리: `requirements.txt` 참고

## 파일 구성
| 파일 | 역할 |
|---|---|
| `model.py` | 이미지·센서·시계열 피처 생성, 모델 정의, 전진 필터 |
| `train.py` | Train 실험 단위 교차검증으로 설정 선택, Validation 확인, 가중치 저장 |
| `test.py` | 입력 CSV에 들어 있는 행만으로 추론하고 제출 파일 생성 |
| `selfcheck.py` | 미래 잘라내기·실험 분리·행 섞기·빈 이미지 자가 점검 |
| `model.joblib` | 학습된 가중치. 재학습 없이 `test.py`만으로 결과를 재현합니다. |

## 실행 방법
```bash
pip install -r requirements.txt
python train.py --data_dir <battery_data_light 경로> --model_out model.joblib --cache_dir ./cache [--refit_train_val]
python test.py --input <battery_data_light 경로>/test/test.csv \
               --image_root <battery_data_light 경로>/test \
               --model model.joblib --output submission.csv \
               --sample_submission <battery_data_light 경로>/sample_submission.csv
python selfcheck.py --data_dir <battery_data_light 경로> --model model.joblib
```

## 학습·검증 구성
- 모델: LightGBM 다중 분류 (300 trees, lr 0.05, num_leaves 31, subsample 0.8, colsample 0.5)
- 학습 데이터: <Train만 / Train+Validation(--refit_train_val)>
- 설정 선택: Train 실험 단위 GroupKFold(5)의 Macro F1 (클래스 가중치 none vs balanced). Validation 은 선택에 사용하지 않음
- 관측 간격 대응: Test 에 10초·15초 간격이 있어, Train 을 2칸·3칸씩 솎아낸 복제본(피처를 솎아낸 행만으로 다시 계산)을 함께 학습
- 검증 점수: <Validation Macro F1 (5초 / 15초 간격)>

## 사용 입력
- 센서: meas_v, meas_t1, meas_t2, meas_p (meas_a 는 상수라 제외), 가스 센서(meas_* 나머지), thermal_t
- 파생: t1−t2, 가스 센서 합, 가스 센서가 동시에 0인 비율
- 이미지: 흑백 밝기 통계 6개 + 색상 구간(어두움·흰색·회색·색상 12구간) 픽셀 비율 + 고온색 비율. 이미지 크기·종횡비는 입력으로 쓰지 않음
- 실험 조건 컬럼(bat_class, bat_cc, soc_status, tr_method)은 사용하지 않음 <운영진 확인 결과에 따라 수정>

## 시계열 피처 구성
- 같은 experiment_id 안에서 relative_time_s 기준 과거 30·90·180초 구간 (t−w, t]의 평균·최댓값·최솟값, w초 전 관측 대비 초당 변화율
- 현재 시점까지의 누적 최댓값·최솟값 (현재 이전·현재 관측만 사용)
- 미래 시점 관측, 실험 전체 기준 통계(전체 평균·최댓값·길이·순위), 다른 실험 관측은 사용하지 않음
- relative_time_s 원값은 입력으로 사용하지 않음
- 전진 필터(선택): 단계는 되돌아가지 않는다는 성질을 이용해 과거 신념을 전이행렬로 전파 후 현재 확률과 결합. 미래를 쓰지 않으며 실험마다 초기화. <사용 여부: train.py 출력 참고>

## 사용한 사전학습 모델
- 없음
