# v2 — Fusion + Sensor Ensemble

## Architecture
- Fusion NN (ResNet18 + Sensor MLP)
- Sensor HistGradientBoostingClassifier
- Soft Voting (50:50)

## Evaluation
- 5-Fold OOF Macro F1: 0.858031378
- DAKER Public Macro F1: 0.8033995479

## Source
- train_sensor_ensemble.py
- test_ensemble.py
- test_fusion.py
- features/

## Weights
weights/ensemble_v1/

## Notes
Image ablation changed 14.38% of OOF predictions.
