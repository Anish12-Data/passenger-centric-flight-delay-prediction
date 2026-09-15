# Passenger-Centric Flight Delay Prediction

Predicting flight delays with a focus on passenger impact — combining connection-buffer and delay-propagation features with classical ML models to flag high-risk flights before they disrupt a passenger's itinerary.

## Project structure

```
flight_disruption_project/
├── data/
│   ├── raw/            # Source data (airlines, airports, cancellation codes; flights.csv excluded — see below)
│   └── processed/       # Engineered feature sets (excluded — see below)
├── notebook/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb # Connection-buffer & propagation features
│   └── 03_experiments.ipynb         # Model training & evaluation
└── results/
    ├── delay_by_airline.png
    ├── delay_by_day.png
    ├── delay_distribution.png
    ├── feature_importance.png
    ├── roc_curves.png
    ├── shap_summary.png
    └── experiment_results.csv
```

## Data

Raw flight-level data (`data/raw/flights.csv`) and the engineered feature set (`data/processed/processed_flights.csv`) are excluded from this repo due to GitHub's file size limits (500MB+ each). The small reference tables (`airlines.csv`, `airports.csv`, `cancellation_codes.csv`) are included. Run `01_eda.ipynb` and `02_feature_engineering.ipynb` to regenerate the processed data from the raw source.

## Experiments

Five experiments progressively add features on top of a logistic regression baseline, then compare model families:

| Experiment | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| E1 Baseline | Logistic Regression | 0.9149 | 0.3592 | 0.7128 | 0.4777 | 0.8653 |
| E2 + ConnBuffer | Logistic Regression | 0.9831 | 0.9607 | 0.7195 | 0.8228 | 0.9088 |
| E3 + Propagation | Logistic Regression | 0.9828 | 0.9542 | 0.7195 | 0.8204 | 0.9093 |
| E4 Full Features | Logistic Regression | 0.9780 | 0.8514 | 0.7242 | 0.7826 | 0.9168 |
| E5 Model Comparison | Random Forest | 0.9840 | 0.9431 | 0.7518 | 0.8366 | 0.9902 |
| E5 Model Comparison | XGBoost | 0.9539 | 0.5434 | 0.9780 | 0.6986 | 0.9924 |

Full results (with standard deviations) in [`results/experiment_results.csv`](results/experiment_results.csv).

Adding connection-buffer and delay-propagation features gives the largest jump in precision and F1 over the raw baseline. Random Forest and XGBoost both push ROC-AUC above 0.99 in the final model comparison, with XGBoost favoring recall and Random Forest favoring precision.

## Reproducing

1. Place raw source files in `data/raw/` (see `.gitignore` for the excluded large file).
2. Run the notebooks in order: `01_eda.ipynb` → `02_feature_engineering.ipynb` → `03_experiments.ipynb`.
3. Outputs (charts, metrics) are written to `results/`.
