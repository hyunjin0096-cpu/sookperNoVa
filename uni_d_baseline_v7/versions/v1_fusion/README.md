# v1 — Fusion NN

## Architecture
- ResNet18 Image Encoder (160x160)
- Sensor MLP
- Feature Concatenation
- 6-Class Classification

## Evaluation
- 5-Fold OOF Macro F1: 0.822819758
- DAKER Public Macro F1: 0.7533384075

## Source
- train_fusion.py
- test_fusion.py
- fusion_model.py
- experiments/
- features/

## Weights
weights/fusion_v1/

## Notes
ResNet18 was trained with random initialization.
