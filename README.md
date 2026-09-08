# Smartphone Addiction Classifier

Machine-learning pipeline for predicting smartphone-addiction labels in a Kaggle-style binary-classification competition.

The project uses an XGBoost classifier and optimizes for ROC-AUC, producing a probability that each person belongs to the positive class rather than only a hard 0/1 prediction.

## Current champion

| Model | Missing-value strategy | Validation ROC-AUC |
| --- | --- | ---: |
| XGBoost | Native NaN handling | **0.964272** |

The champion configuration is defined in [`main.py`](main.py) and uses a reproducible random seed of `42`.

## Dataset

Place the competition files in `Dataset/`:

```text
Dataset/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Expected columns:

- `train.csv`: `id`, predictive features, and binary target `addicted_label`
- `test.csv`: `id` and the same predictive features, without `addicted_label`
- `sample_submission.csv`: `id` and `addicted_label`

The dataset is intentionally excluded from Git because competition data is large and may be subject to the competition's terms. Download it from the relevant Kaggle competition and keep it local.

## Project structure

```text
.
├── main.py                    # Main entry point and experiment switches
├── src/
│   ├── config.py              # Paths, target column, and reproducibility settings
│   ├── data_loader.py         # CSV loading, schema validation, and splitting
│   ├── data_analysis.py       # Exploratory and missingness analysis
│   ├── preprocessing.py       # Feature preparation and missing-value strategies
│   ├── xgboost_model.py       # XGBoost pipeline construction
│   ├── model_evaluation.py    # Holdout evaluation with ROC-AUC
│   ├── cross_validation.py    # Stratified cross-validation workflow
│   ├── benchmark_models.py    # Baseline model experiments
│   └── hyperparameter_tuning.py
└── Dataset/                   # Local, ignored competition files
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn xgboost
```

## Run

After placing the CSV files in `Dataset/`, run:

```bash
python main.py
```

By default, the script validates the competition data, creates a stratified 80/20 holdout split, and evaluates the current champion model. Optional workflows can be enabled by changing the `RUN_*` flags near the top of `main.py`:

- `RUN_GENERAL_EDA` — general exploratory analysis
- `RUN_MISSINGNESS_EDA` — missing-value analysis
- `RUN_XGBOOST_STRATEGY_EXPERIMENTS` — compare imputation strategies
- `RUN_CROSS_VALIDATION` — run five-fold stratified cross-validation
- `RUN_HYPERPARAMETER_TUNING` — start the tuning pipeline

## Methodology

1. Load and validate `train.csv`, `test.csv`, and `sample_submission.csv`.
2. Separate identifiers, features, and the binary target.
3. Preserve missing values for the champion XGBoost configuration.
4. Train a histogram-based XGBoost classifier with the competition metric set to AUC.
5. Evaluate using predicted positive-class probabilities and `roc_auc_score`.

ROC-AUC is the primary metric because the competition evaluates ranking quality across classification thresholds. Accuracy and hard labels are therefore not used for model selection.

## Reproducibility

The pipeline uses `RANDOM_STATE = 42` for the train/validation split and model initialization. Results can still vary with different library versions or hardware. Record the Python and package versions when comparing experiments.

## Notes

- Do not commit competition data, generated submissions, virtual environments, or model binaries.
- The test set must not be used for model selection.
- A competition submission should contain the columns `id,addicted_label`, with probability values in `addicted_label`.

## License

No license has been selected for this project yet. Add a `LICENSE` file before accepting external contributions or defining reuse terms.
