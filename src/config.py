from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"

TRAIN_PATH = DATASET_DIR / "train.csv"
TEST_PATH = DATASET_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATASET_DIR / "sample_submission.csv"

TARGET_COLUMN = "addicted_label"
ID_COLUMN = "id"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

PAIRWISE_MIN_SUPPORT = 500