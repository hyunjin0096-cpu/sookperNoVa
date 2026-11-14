from pathlib import Path

ROOT = Path(__file__).resolve().parent
# 압축을 푼 데이터 폴더를 프로젝트의 data/battery_data_light 에 배치하세요.
DATA_DIR = ROOT / 'data' / 'battery_data_light'
ARTIFACT_DIR = ROOT / 'artifacts'  # 예전 학습 결과와 호환용
WEIGHTS_DIR = ROOT / 'weights'
CACHE_DIR = ROOT / 'cache'
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32  # GPU 메모리가 부족하면 8 또는 16으로 낮추세요.
CV_FOLDS = 5
# 데이터에 존재하며 추론 시에도 이용 가능한 컬럼만 여기에 기재합니다.
EXTRA_NUMERIC_COLUMNS = [
    'thermal_t', 'indu_t', 'tr_method', 'bat_class',
    'bat_capacity', 'rated_capacity', 'capacity_ah', 'soc', 'SOC',
]
# 가스 센서는 이름이 gas 로 시작하는 컬럼만 자동 포함합니다.
# 다른 이름이라면 CSV 헤더를 보고 위 리스트에 명시적으로 추가하세요.
GAS_PREFIXES = ('gas', 'meas_gas')
