# Multi-Objective Student Performance Prediction with XAI and Recommendation System

A research pipeline for predicting student academic outcomes (CGPA category, placement status) using Explainable AI and generating personalised improvement recommendations. Compares **7 models** (4 baselines + 3 attention-based deep learning) with SHAP-based explainability.

## Models

- **Baselines**: Logistic Regression, Random Forest, SVM, XGBoost
- **Deep Learning**: TabNet, FT-Transformer, SAINT

## Objectives

1. Multi-objective performance prediction using 7 ML/DL models
2. Feature engineering with composite scores, interaction features, and low-variance removal
3. Optuna-based hyperparameter tuning framework
4. Explainability via SHAP values and attention masks (TabNet, FT-Transformer, SAINT)
5. SHAP-based recommendation system targeting modifiable student features
6. Comparative validation against baselines and literature
7. Publication-ready visualizations and research paper draft

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python -m src.pipeline --stage preprocess
python -m src.pipeline --stage baselines
python -m src.pipeline --stage dlmodels    # TabNet + FT-Transformer + SAINT
python -m src.pipeline --stage shap
python -m src.pipeline --stage recommend
python -m src.pipeline --stage visualize
python -m src.pipeline --stage paper

# Optional: Optuna hyperparameter tuning (run before baselines/dlmodels)
python -m src.pipeline --stage tune

# Run tests
python -m pytest tests/ -v
```

## Project Structure

```
├── dataset/                  # Student dataset CSV
├── src/
│   ├── config.py             # Central configuration
│   ├── pipeline.py           # CLI orchestrator
│   ├── data/                 # Data loading, preprocessing & feature engineering
│   ├── models/               # Baselines, TabNet, FT-Transformer, SAINT, tuning
│   ├── explainability/       # SHAP analysis (global & local)
│   ├── recommendations/      # SHAP-based recommendation engine
│   ├── evaluation/           # Metrics & model comparison
│   ├── visualization/        # Publication-quality plots
│   └── paper/                # Research paper draft generator
├── tests/                    # Unit & integration tests
├── results/                  # Generated outputs (gitignored)
│   ├── models/               # Saved model files
│   ├── shap/                 # SHAP values, importance & attention CSVs
│   ├── tables/               # Comparison tables & result CSVs
│   ├── plots/                # Visualization PNGs (26+ plots)
│   ├── reports/              # Recommendation & literature reports
│   └── paper/                # Paper draft with figures & tables
└── requirements.txt
```

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| `preprocess` | Load CSV, clean data, feature engineering, encode features, create targets |
| `tune` | *(Optional)* Optuna hyperparameter optimization for baselines and TabNet |
| `baselines` | Train LR, RF, SVM, XGBoost with 5-fold stratified CV + SMOTE |
| `dlmodels` | Train TabNet, FT-Transformer, SAINT with CV and attention extraction |
| `shap` | Compute SHAP values for all models, cross-model comparison |
| `recommend` | Generate per-student recommendations from SHAP values |
| `visualize` | Create comparison plots, SHAP summaries, attention heatmaps |
| `paper` | Generate structured Markdown research paper draft |

## Key Configuration

- **Random seed**: 42 (reproducibility)
- **CV folds**: 5 (stratified)
- **CGPA thresholds**: Low < 6.5, Medium 6.5–8.0, High > 8.0
- **Primary metric**: F1-macro
- **Class imbalance**: SMOTE on training folds only
- **Feature engineering**: Composite academic/skills/engagement scores, interaction features

## Requirements

- Python 3.10 or 3.11
- PyTorch, pytorch-tabnet, SHAP, scikit-learn, XGBoost, imbalanced-learn, Optuna
