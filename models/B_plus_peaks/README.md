# Battery Thermal Runaway Classification — B + Peaks

## Model
- Architecture: LightGBM
- Base: Teammate V3-B (190 features)
- Added: 5 physics-informed peak-history features
- Total features: 195
- Class weight: balanced
- Monotonic postprocessing: disabled

## Performance
| Metric | Baseline B | B + Peaks |
|---|---:|---:|
| Train GroupKFold Macro F1 | 0.8933 | 0.8964 |
| Validation Macro F1 | 0.8123 | 0.8151 |
| Validation Accuracy | 0.9024 | 0.9039 |
| Stage 5 F1 | 0.6907 | 0.6958 |
| Stage 6 F1 | 0.8926 | 0.8993 |
| Stage 5 -> 4 errors | 472 | 470 |

## Training

Run from this directory:

python experiments/v3b_physics/train_physics.py \
  --data-dir /path/to/competition_data \
  --groups peaks \
  --output-dir artifacts/v3b_physics

## Notes
- Five-fold GroupKFold by experiment ID.
- Causal features use current and past observations only.
- Uses original competition data and thermal images during training.
- Private datasets and image caches are not included.
- Based on teammate V3-B; extended with peak-history features.
