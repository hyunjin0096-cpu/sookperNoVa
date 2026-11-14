
from pathlib import Path
import json
import joblib
import numpy as np

from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix

from config import WEIGHTS_DIR, SEED
from data_utils import read_split
from image_encoder import load_encoder
from train import get_features

LABELS = list(range(6))


def macro_f1(y, pred):
    return f1_score(
        y, pred, labels=LABELS,
        average="macro", zero_division=0
    )


def new_model(iterations=1000):
    return CatBoostClassifier(
        iterations=iterations,
        depth=6,
        learning_rate=0.03,
        loss_function="MultiClass",
        l2_leaf_reg=5,
        random_seed=SEED,
        thread_count=4,
        verbose=False,
        allow_writing_files=False
    )


def main():
    # 기존 베이스라인 모델 불러오기
    base_dir = Path(WEIGHTS_DIR)
    baseline = joblib.load(base_dir / "lightgbm.joblib")

    with open(base_dir / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    # 데이터 및 이미지 특징 불러오기
    train = read_split("train")
    validation = read_split("validation")

    y = train.label.astype(int).to_numpy() - 1
    y_val = validation.label.astype(int).to_numpy() - 1

    encoder, device = load_encoder(
        str(base_dir / "resnet18.pth")
    )

    extra = meta["extra_columns"]

    X = get_features(
        "train", train, encoder, device, extra,
        tabular_only=meta["tabular_only"]
    )
    X_val = get_features(
        "validation", validation, encoder, device, extra,
        tabular_only=meta["tabular_only"]
    )

    X = X.reindex(columns=meta["features"])
    X_val = X_val.reindex(columns=meta["features"])

    # 실험별 GroupKFold
    cv = GroupKFold(n_splits=5)
    scores = []
    best_rounds = []

    for i, (a, b) in enumerate(
        cv.split(X, y, groups=train.experiment_id), 1
    ):
        model = new_model()

        model.fit(
            X.iloc[a], y[a],
            eval_set=(X.iloc[b], y[b]),
            early_stopping_rounds=60,
            verbose=False
        )

        pred = model.predict(X.iloc[b]).astype(int).ravel()
        score = macro_f1(y[b], pred)

        scores.append(score)
        best_rounds.append(model.best_iteration_ + 1)

        print(f"Fold {i}: Macro F1={score:.4f}")

    print("\nCatBoost CV:", np.mean(scores))
    print("CV 표준편차:", np.std(scores))

    # 전체 Train으로 최종 학습
    trees = max(100, int(np.median(best_rounds)))
    model = new_model(iterations=trees)
    model.fit(X, y)

    p_cat = model.predict_proba(X_val)
    p_base = baseline.predict_proba(X_val)

    assert np.array_equal(model.classes_, np.arange(6))
    assert np.array_equal(baseline.classes_, np.arange(6))

    # 단독 성능
    pred_cat = np.argmax(p_cat, axis=1)
    pred_base = np.argmax(p_base, axis=1)

    print("\n=== 단독 성능 ===")
    print("Baseline:", macro_f1(y_val, pred_base))
    print("CatBoost:", macro_f1(y_val, pred_cat))

    # 앙상블 성능
    print("\n=== 앙상블 성능 ===")

    for w in [0.25, 0.5, 0.75]:
        combined = (1-w)*p_base + w*p_cat
        pred = np.argmax(combined, axis=1)

        print(
            f"CatBoost 비율 {w:.0%}:",
            macro_f1(y_val, pred)
        )

    # 오분류 비교
    print("\n=== 주요 오분류 비교 ===")

    for name, pred in [
        ("Baseline", pred_base),
        ("CatBoost", pred_cat),
        ("Ensemble 50:50",
         np.argmax(0.5*p_base + 0.5*p_cat, axis=1))
    ]:
        wrong_5to4 = np.sum((y_val == 4) & (pred == 3))
        wrong_3to2 = np.sum((y_val == 2) & (pred == 1))

        print(
            f"{name}: "
            f"5->4 = {wrong_5to4}, "
            f"3->2 = {wrong_3to2}"
        )

    print("\n=== CatBoost Confusion Matrix ===")
    print(confusion_matrix(
        y_val + 1, pred_cat + 1,
        labels=list(range(1, 7))
    ))


if __name__ == "__main__":
    main()
