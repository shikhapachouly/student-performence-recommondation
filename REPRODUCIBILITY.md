# Reproducibility Guide

This document provides detailed instructions to reproduce all tables and figures from the paper:
**"Explainable AI for Student Success: A Multi-Objective Framework with SHAP-Based Personalized Recommendations"**

## Quick Start

To reproduce all results with a single command:

```bash
python reproduce_paper.py
```

This will run the complete pipeline and generate all tables/figures in the `results/` directory.

## Environment Setup

### Option 1: Using pip (Recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
# Create conda environment
conda create -n student-performance python=3.10
conda activate student-performance

# Install dependencies
pip install -r requirements.txt
```

## Reproducibility Checklist

### 1. Random Seeds
- **Global seed**: `RANDOM_SEED = 42` (src/config.py:23)
- Applied to: NumPy, scikit-learn, PyTorch, Optuna, SMOTE
- Ensures deterministic results across runs

### 2. Cross-Validation Splits
- **Method**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Fold indices**: Saved to `results/folds/fold_indices.json` after first run
- Identical splits used for all models and objectives

### 3. SMOTE Configuration
- **Applied**: Within training folds only (no data leakage)
- **Parameters**: `SMOTE(random_state=42, k_neighbors=5)`
- **Location**: src/models/baselines.py:181

### 4. Optuna Hyperparameter Tuning
- **Trials**: 50 per model
- **Inner CV**: 3-fold
- **Sampler**: TPE with seed=42
- **Search spaces**: src/config.py:352-430
- **Best parameters**: Saved to `results/hyperparameters/best_params.json`

## Reproducing Specific Tables

### Table 2: Leakage Control Ablation
```bash
python -m src.review_comments.revision.code.leakage_ablation
```
Output: `src/review_comments/revision/output/tables/leakage_ablation.csv`

### Table 4: Feature Categories
Already in manuscript; counts from preprocessing:
```bash
python -m src.pipeline --stage preprocess
```

### Table 6: Deep Learning Configuration
Static configuration in manuscript; values from src/config.py:334-350

### Table 7: Complete Model Performance
```bash
python -m src.pipeline --stage baselines
python -m src.pipeline --stage dlmodels
```
Output: `results/tables/model_comparison.csv`

### Table 9: SOTA Comparison
```bash
python -m src.review_comments.revision.code.sota_baselines
```
Output: `src/review_comments/revision/output/tables/sota_comparison.csv`

### Table 10: PSM Analysis
```bash
python -m src.review_comments.revision.code.causal_psm
```
Output: `src/review_comments/revision/output/tables/psm_results.csv`

### Table 11: Personalized Recommendations
```bash
python -m src.pipeline --stage recommend
```
Output: `results/reports/recommendations_*.json`

### Table A1: Optuna Search Spaces
```bash
python -m src.review_comments.revision.code.replication_tables
```
Output: `src/review_comments/revision/output/tables/tableA1_search_spaces.md`

## Reproducing Specific Figures

### Figure 1: Model Comparison for CGPA
```bash
python -m src.pipeline --stage visualize
```
Output: `results/plots/model_comparison_cgpa.png`

### Figure 2: Model Comparison for At-Risk
```bash
python -m src.pipeline --stage visualize
```
Output: `results/plots/model_comparison_at_risk.png`

### Figure 3-5: SHAP Summary Plots
```bash
python -m src.pipeline --stage shap
python -m src.pipeline --stage visualize
```
Output: `results/plots/shap_summary_*.png`

### Figure 6-7: SHAP Waterfall Plots
```bash
python -m src.pipeline --stage shap
python -m src.pipeline --stage visualize
```
Output: `results/plots/shap_waterfall_*.png`

### Figure 8: Feature Importance by Category
```bash
python -m src.pipeline --stage visualize
```
Output: `results/plots/feature_importance_by_category.png`

### Figure 9: TabNet Attention Heatmap
```bash
python -m src.pipeline --stage dlmodels
python -m src.pipeline --stage visualize
```
Output: `results/plots/tabnet_attention_heatmap.png`

### Figure 13: Fairness Audit
```bash
python -m src.review_comments.revision.code.fairness
```
Output: `src/review_comments/revision/output/plots/fairness_audit.png`

## Dataset Information

### Original Dataset
- **Location**: `dataset/student-dataset.csv`
- **Size**: 597KB
- **Records**: 1,378 students
- **Features**: 48 (after preprocessing)
- **Privacy**: Anonymized student IDs

### Synthetic Dataset (Privacy-Preserving Alternative)
To generate a synthetic dataset with similar statistical properties:

```bash
python -m src.data.generate_synthetic
```

This creates `dataset/synthetic_student_dataset.csv` that:
- Preserves statistical distributions
- Maintains feature correlations
- Removes any identifying information
- Can be shared publicly

## Verification

To verify that your reproduction matches the paper results:

```bash
python -m src.review_comments.revision.code.verify_final
```

This will check:
- Model performance metrics (within ±0.01 tolerance)
- SHAP value consistency
- Table values match manuscript
- Figure generation completeness

## Common Issues

### Issue: Different results across runs
**Solution**: Ensure all random seeds are set correctly. Check that CUDA deterministic mode is enabled for PyTorch models.

### Issue: Memory errors with deep learning models
**Solution**: Reduce batch size in src/config.py (TABNET_BATCH_SIZE, FT_BATCH_SIZE, SAINT_BATCH_SIZE)

### Issue: Missing dependencies
**Solution**: Ensure you're using Python 3.10 or 3.11. Some packages may have version conflicts with Python 3.12+.

## Contact

For questions about reproduction:
- Open an issue on GitHub: https://github.com/shikhapachouly/student-recommondation/issues
- Email: shikhaphd2022@gmail.com

## Citation

If you use this code or data, please cite:

```bibtex
@article{pachouly2025explainable,
  title={Explainable AI for Student Success: A Multi-Objective Framework with SHAP-Based Personalized Recommendations},
  author={Pachouly, Shikha and Bormane, D.S.},
  journal={[Journal Name]},
  year={2025}
}
```
