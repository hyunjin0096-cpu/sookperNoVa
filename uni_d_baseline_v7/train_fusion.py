"""Final-submission training entrypoint (do not run during inference).

After model selection, this may replace train.py in the submission repository.
It delegates to the full train/CV implementation in experiments/train_fusion.py.
"""
from experiments.train_fusion import main

if __name__ == "__main__":
    main()
