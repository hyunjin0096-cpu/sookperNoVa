"""Compare baseline 0.6674 on identical original Validation rows (78->19 protocol)."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from cv.common import LABELS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation-csv", required=True)
    p.add_argument("--predictions", required=True, help="CSV: sample_id,pred_label")
    args = p.parse_args()
    gold = pd.read_csv(args.validation_csv)[["sample_id", "label"]]
    pred = pd.read_csv(args.predictions)
    if not {"sample_id", "pred_label"}.issubset(pred.columns):
        raise ValueError("predictions need sample_id,pred_label")
    if pred.sample_id.duplicated().any() or set(gold.sample_id) != set(pred.sample_id):
        raise ValueError("Must predict each and every original Validation row once")
    df = gold.merge(pred[["sample_id", "pred_label"]], on="sample_id", validate="one_to_one")
    if not df.pred_label.isin(LABELS).all():
        raise ValueError("pred_label must be 1..6")
    print("Holdout Macro F1:", f1_score(df.label, df.pred_label, labels=LABELS,
                                         average="macro", zero_division=0))
    print("Prior baseline (same split, if training uses ONLY Train 78): 0.6674")
    print(classification_report(df.label, df.pred_label, labels=LABELS, zero_division=0))
    print("WARNING: Metric tool cannot prove model training excluded Validation. Verify training provenance!")


if __name__ == "__main__":
    main()
