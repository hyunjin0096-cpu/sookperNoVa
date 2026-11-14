
import json
import hashlib
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from config import WEIGHTS_DIR
from data_utils import read_split, image_roots
from image_encoder import load_encoder, extract
from features import fuse


# 1. 저장된 두 모델 경로
BASE_DIR = Path(WEIGHTS_DIR)
TUNE_DIR = Path(__file__).resolve().parent / "tuning_num_leaves_15"


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_saved_model(folder):
    model = joblib.load(folder / "lightgbm.joblib")

    with open(folder / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    return model, meta


def make_features(frame, embeddings, stats, meta):
    X = fuse(
        frame,
        embeddings,
        stats,
        meta["extra_columns"],
        use_embeddings=not meta["tabular_only"]
    )

    missing = set(meta["features"]) - set(X.columns)
    if missing:
        raise ValueError(f"필요한 특징이 없습니다: {missing}")

    return X.reindex(columns=meta["features"])


def macro_f1(y, pred):
    return f1_score(
        y, pred,
        labels=list(range(6)),
        average="macro",
        zero_division=0
    )


def main():
    # 2. 모델 불러오기
    base_model, base_meta = load_saved_model(BASE_DIR)
    tune_model, tune_meta = load_saved_model(TUNE_DIR)

    print("베이스라인 모델:", BASE_DIR)
    print("Tune 1 모델:", TUNE_DIR)

    # 3. 이미지 인코더가 동일한지 확인
    base_weights = BASE_DIR / "resnet18.pth"
    tune_weights = TUNE_DIR / "resnet18.pth"

    if file_hash(base_weights) != file_hash(tune_weights):
        raise ValueError(
            "두 모델의 ResNet18 가중치가 다릅니다. "
            "각각의 가중치로 이미지 특징을 추출해야 합니다."
        )

    # 4. Validation 데이터 불러오기
    validation = read_split("validation")
    y_val = validation["label"].astype(int).to_numpy() - 1

    encoder, device = load_encoder(str(base_weights))

    print("이미지 특징 추출 중...")
    embeddings, stats = extract(
        validation,
        image_roots("validation"),
        encoder,
        device
    )

    X_base = make_features(
        validation, embeddings, stats, base_meta
    )
    X_tune = make_features(
        validation, embeddings, stats, tune_meta
    )

    # 5. 각 모델의 클래스별 예측 확률
    p_base = base_model.predict_proba(X_base)
    p_tune = tune_model.predict_proba(X_tune)

    assert np.array_equal(base_model.classes_, np.arange(6))
    assert np.array_equal(tune_model.classes_, np.arange(6))

    # 6. 단독 모델 성능 비교
    pred_base = np.argmax(p_base, axis=1)
    pred_tune = np.argmax(p_tune, axis=1)

    print("\n=== 단독 모델 성능 ===")
    print("Baseline F1:", round(macro_f1(y_val, pred_base), 4))
    print("Tune 1 F1:", round(macro_f1(y_val, pred_tune), 4))

    # 7. 앙상블 비율별 성능 비교
    print("\n=== 앙상블 결과 ===")

    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # w는 Tune 1의 가중치
        combined = (1 - w) * p_base + w * p_tune
        pred = np.argmax(combined, axis=1)

        score = macro_f1(y_val, pred)

        print(
            f"Baseline {1-w:.2f} + Tune1 {w:.2f}"
            f" -> Macro F1: {score:.4f}"
        )

    # 8. 50:50 앙상블 상세 분석
    combined = 0.5 * p_base + 0.5 * p_tune
    pred = np.argmax(combined, axis=1)

    print("\n=== 50:50 앙상블 클래스별 성능 ===")
    print(classification_report(
        y_val + 1,
        pred + 1,
        labels=list(range(1, 7)),
        zero_division=0
    ))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(
        y_val + 1,
        pred + 1,
        labels=list(range(1, 7))
    ))

    # 9. 클래스 5 -> 4, 클래스 3 -> 2 오분류 비교
    print("\n=== 주요 오분류 비교 ===")

    for name, predictions in [
        ("Baseline", pred_base),
        ("Tune 1", pred_tune),
        ("Ensemble", pred)
    ]:
        wrong_5to4 = np.sum(
            (y_val == 4) & (predictions == 3)
        )
        wrong_3to2 = np.sum(
            (y_val == 2) & (predictions == 1)
        )

        print(
            f"{name}: 5->4 = {wrong_5to4}개, "
            f"3->2 = {wrong_3to2}개"
        )


if __name__ == "__main__":
    main()
