from pathlib import Path
import pandas as pd
from config import DATA_DIR


def read_split(split, data_dir=DATA_DIR):
    path = Path(data_dir) / split / f'{split}.csv'
    if not path.exists():
        raise FileNotFoundError(f'{path} 가 없습니다. README의 데이터 폴더 구조를 확인하세요.')
    df = pd.read_csv(path, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    for c in ['sample_id','experiment_id','relative_time_s','image_path']:
        if c not in df.columns:
            raise ValueError(f'{split}.csv에 {c} 컬럼이 없습니다.')
    if df.sample_id.isna().any() or df.sample_id.duplicated().any():
        raise ValueError(f'{split} sample_id 누락 또는 중복')
    return df


def image_roots(split, data_dir=DATA_DIR):
    d = Path(data_dir)
    return [d / split, d]
