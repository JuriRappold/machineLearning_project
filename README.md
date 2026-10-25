# Hourly Electricity Load Prediction

A machine learning course project comparing:
- **Linear regression**, implemented from scratch with NumPy and gradient descent.
- **Random forest regression**, implemented with scikit-learn.

The target is electricity load in **MW**. See `report.pdf` for results and limitations; `report.tex` contains the LaTeX source.

## Setup

Use Python 3.12 and run commands from the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirments.txt
```

## Data

Source: [Hourly Energy Consumption on Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption).

Both training workflows read `data/all_data_READY.csv`, containing:
- `Company`: company/region identifier.
- `Datetime`: hourly timestamp.
- `MW`: observed electricity load.

The prepared dataset contains 933,922 observations across ten groups. Preparation removes duplicate timestamps, missing values, invalid hourly timestamps, incomplete days, and pooled Z-score outliers. PJME, PJMW, and PJM_Load are combined into PJM, averaging overlapping readings.

## Train the Models

Ensure the prepared CSV is present, then run:

```bash
mkdir -p figures/statistics

# Linear regression
python -c "from training_model import train_linea_regression; train_linea_regression()"

# Random forest (hyperparameter tuning can take considerable time)
python -c "from random_forests import train; train()"
```

Training evaluates the models and overwrites their saved outputs:
- `lr_gradient_energy_model.joblib`
- `rf_energy_model.joblib`
- Evaluation figures in `figures/statistics/`.

The forest model is approximately 423 MiB before compression. 
## Main Files

| File | Purpose |
|---|---|
| `clean.py`, `split_and_merge.py`, `relabel_companies.py` | Initial data preparation |
| `average_duplicates_after_joining_of_PJM.py`, `remove_outliers.py` | PJM averaging and outlier filtering |
| `feature_extraction.py` | Linear-model features, encoding, scaling, and chronological split |
| `linear_regression.py` | From-scratch regression and evaluation metrics |
| `training_model.py` | Linear regression training |
| `random_forests.py` | Forest features, tuning, training, and evaluation |
| `main.py` | Preparation followed by training both models |

**Note:** `main.py` expects existing intermediate data. Initial cleaning, merging, and relabeling calls are disabled in `data_preparation.py`; it is not a complete raw-data rebuild command.

## Evaluation Notes

Models use chronological train/test splits and MAE, RMSE, and R². The forest uses historical-load features, while linear regression does not, and their test samples differ. Gaps affect lag timing, and the linear time-trend feature currently resets between training and testing. These limitations are discussed in the report.
