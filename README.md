# Multi-Objective Student Performance Prediction with XAI and Recommendation System

A research pipeline for predicting student academic outcomes (CGPA category, placement status) using Explainable AI and generating personalised improvement recommendations. Compares **7 models** (4 baselines + 3 attention-based deep learning) with SHAP-based explainability.

📄 **Paper**: "Explainable AI for Student Success: A Multi-Objective Framework with SHAP-Based Personalized Recommendations"

## 🔄 Reproducibility

**One-command reproduction of all results:**
```bash
python reproduce_paper.py
```

For detailed reproducibility instructions, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)

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

# Reproduce all results with one command
python reproduce_paper.py
```

## Environment Setup

### Option 1: pip (Recommended)
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Option 2: Conda
```bash
conda create -n student-performance python=3.10
conda activate student-performance
pip install -r requirements.txt
```

### Dependencies (`requirements.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0.0,<2.5.0 | Deep learning backend (TabNet, FT-Transformer, SAINT) |
| pytorch-tabnet | >=4.0,<5.0 | TabNet architecture |
| shap | >=0.44.0,<0.47.0 | SHAP explainability (TreeExplainer, KernelExplainer) |
| scikit-learn | >=1.3.0,<1.6.0 | ML baselines, StratifiedKFold, metrics |
| imbalanced-learn | >=0.11.0,<0.13.0 | SMOTE class-imbalance handling |
| xgboost | >=2.0.0,<2.2.0 | XGBoost primary model |
| optuna | >=3.0.0 | Bayesian hyperparameter tuning (TPE sampler) |
| numpy | >=1.24.0,<2.0.0 | Numerical computing |
| pandas | >=2.0.0,<2.3.0 | Data manipulation |
| matplotlib | >=3.7.0,<3.10.0 | Visualization |
| seaborn | >=0.13.0,<0.14.0 | Statistical plots |
| scipy | >=1.11.0,<1.14.0 | Statistical functions |
| pytest | latest | Unit testing |

## Reproducing Manuscript Tables and Figures

### One-Command Reproduction

```bash
python reproduce_paper.py
```

### Step-by-Step Commands

| Step | Command | Manuscript Output |
|------|---------|-------------------|
| 1. Preprocess | `python -m src.pipeline --stage preprocess` | Dataset cleaning, feature engineering, target creation |
| 2. Tune | `python -m src.pipeline --stage tune` | Optuna hyperparameter optimization (50 trials × 3 inner CV) |
| 3. Baselines | `python -m src.pipeline --stage baselines` | Table 7: LR, RF, SVM, XGBoost results |
| 4. Deep Learning | `python -m src.pipeline --stage dlmodels` | Table 7: TabNet, FT-Transformer, SAINT results |
| 5. SHAP | `python -m src.pipeline --stage shap` | Figures 3–8: SHAP summary, waterfall, feature importance |
| 6. Recommend | `python -m src.pipeline --stage recommend` | Table 11: Personalized recommendations |
| 7. Visualize | `python -m src.pipeline --stage visualize` | Figures 1–2, 9: Model comparison, attention heatmap |
| 8. Paper | `python -m src.pipeline --stage paper` | Structured Markdown paper draft |
| 9. Tests | `python -m pytest tests/ -v` | Unit and integration tests |

### Revision-Specific Tables

| Manuscript Table | Command | Output File |
|-----------------|---------|-------------|
| Table 3: Leakage Control | `python -m src.review_comments.revision.code.leakage_ablation` | `revision/output/tables/leakage_ablation.csv` |
| Table 9: SOTA Comparison | `python -m src.review_comments.revision.code.sota_baselines` | `revision/output/tables/sota_comparison.csv` |
| Table 10: PSM Analysis | `python -m src.review_comments.revision.code.causal_psm` | `revision/output/tables/psm_results.csv` |
| Table A1: Optuna Search Spaces | `python -m src.review_comments.revision.code.replication_tables` | `revision/output/tables/tableA1_search_spaces.md` |
| Table A2: Best Hyperparameters | `python -m src.review_comments.revision.code.replication_tables` | `revision/output/tables/tableA2_best_hyperparams.md` |
| Fairness Audit (Table + Fig. 13) | `python -m src.review_comments.revision.code.fairness` | `revision/output/tables/fairness_*.csv` |

### Verification

```bash
python -m src.review_comments.revision.code.verify_final
```

## Project Structure

```
├── dataset/
│   └── student-dataset.csv       # Anonymized student dataset (1,378 records)
├── src/
│   ├── config.py                 # Central config: seeds, thresholds, Optuna search spaces
│   ├── pipeline.py               # CLI orchestrator
│   ├── data/                     # Preprocessing, feature engineering, synthetic data generator
│   ├── models/                   # Baselines, TabNet, FT-Transformer, SAINT, Optuna tuning
│   ├── explainability/           # SHAP analysis (global & local)
│   ├── recommendations/          # SHAP-based recommendation engine
│   ├── evaluation/               # Metrics & model comparison
│   ├── visualization/            # Publication-quality plots
│   └── review-comments/revision/ # Revision code: SOTA, PSM, fairness, leakage ablation
├── tests/                        # Unit & integration tests
├── results/                      # Generated outputs (gitignored)
├── reproduce_paper.py            # One-command reproduction script
├── requirements.txt              # Python dependencies with version ranges
├── REPRODUCIBILITY.md            # Detailed reproduction instructions
└── README.md
```

## Reproducibility Guarantees

| Parameter | Value | Location |
|-----------|-------|----------|
| Random seed | 42 | `src/config.py:23` |
| CV folds | 5 (stratified) | `src/config.py:24` |
| CV split method | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` | `src/models/baselines.py:172` |
| SMOTE | `SMOTE(random_state=42)`, training folds only | `src/models/baselines.py:181` |
| Optuna trials | 50 per model | `src/config.py` |
| Optuna inner CV | 3-fold | `src/config.py` |
| Optuna sampler | TPE with seed=42 | `src/models/baselines.py` |
| CGPA thresholds | Low < 6.5, Medium 6.5–8.0, High > 8.0 | `src/config.py:27-28` |
| Primary metric | F1-macro | `src/config.py` |
| Feature engineering | Composite academic/skills/engagement scores, interaction features | `src/data/feature_engineering.py` |

## Requirements

- **Python**: 3.10 or 3.11
- **GPU**: Optional (CUDA for faster DL training)
- **All dependencies**: Listed in `requirements.txt` with version ranges
