# Uni-D 배터리 열폭주 6단계 분류 — LightGBM + 확장 피처 (베이스라인 v7 기반)

> 참가팀은 이 README를 자기 팀 내용으로 바꿔서 제출하십시오. `<>` 부분을 채우면 됩니다.

## 팀 정보
- 팀명: <팀명>
- 최종 선택 제출 파일: <리더보드에서 선택한 파일명>, Private 복원 대상

## 개발 환경
- OS: <예: Ubuntu 22.04>
- Python: <예: 3.10>
- GPU: <예: 대회 제공 GPU 서버, 모델명>
- 라이브러리: `requirements.txt` 참고

## 파일 구성
| 파일 | 역할 |
|---|---|
| `model.py` | 입력 컬럼 규격, 피처 생성(이미지 밝기·열 지수 통계, 센서·가스·실험 조건, 과거 방향 시계열), LightGBM 정의 |
| `train.py` | 간격 증강 포함 학습, Train GroupKFold로 설정 선택, Validation(원본·15초 간격)으로 확인, 모델·OOF 확률 저장 |
| `test.py` | 입력 CSV에 들어 있는 행만으로 추론하고 제출 파일 생성 |
| `model.joblib` | 학습된 가중치. 재학습 없이 `test.py`만으로 결과를 재현할 수 있어야 합니다. |

## 실행 방법
```bash
pip install -r requirements.txt

# 학습 (선택: --refit_train_val 을 주면 Train+Validation으로 최종 학습)
#   --augment_steps 2 3  : 10초·15초 간격 증강(기본값). 끄려면 --augment_steps 만 쓰기
#   --exclude <컬럼...>  : 피처에서 뺄 CSV 컬럼
python train.py --data_dir <battery_data_light 경로> --model_out model.joblib

# 추론
python test.py --input <battery_data_light 경로>/test/test.csv \
               --image_root <battery_data_light 경로>/test \
               --model model.joblib --output submission.csv \
               --sample_submission <battery_data_light 경로>/sample_submission.csv
```

## 학습·검증 구성
- 학습 데이터: <예: Train만 / Train+Validation>
- 검증 방법: <예: 제공 Validation + Train 실험 단위 GroupKFold(5)>
- 검증 점수: <Validation Macro F1>

## 시계열 피처 구성
- 같은 experiment_id 안에서 relative_time_s 기준 과거 30·60·120초 구간의 평균·최댓값, 윈도우 이전 관측과의 차이와 초당 기울기
- 직전 관측 대비 초당 기울기, 실험 시작부터 현재까지의 누적 최댓값(전압은 누적 최솟값), 첫 관측 대비 변화량
- 대상 신호: meas_v, meas_t1, meas_t2, meas_p, thermal_t, 이미지 img_p99·열 지수 통계, 가스 센서 0 개수
- Test의 넓은 관측 간격(10·15초)에 대비해, Train 실험을 2·3행 간격으로 솎은 복사본을 학습에 추가(교차검증 그룹은 원래 실험 ID)
- 미래 시점 관측과 실험 전체 기준 통계는 사용하지 않습니다.

## 이미지 피처
- 흑백 밝기 통계 6개(베이스라인과 동일)
- 의사색 HSV 색상을 고정 규칙으로 0~1 열 지수로 변환(자홍·보라=0 → 빨강=0.9, 흰색=1)한 뒤 평균·표준편차·백분위·최댓값·고온 비율
- 이미지 크기·종횡비는 실험 식별 지름길이 될 수 있어 사용하지 않습니다.

## 사용한 사전학습 모델
- <없음, 또는 모델명 / 출처 / 버전 / 라이선스>
