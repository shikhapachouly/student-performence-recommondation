#!/usr/bin/env python
"""Complete pipeline runner: preprocess -> baselines -> DL -> SHAP -> recommend -> visualize.
Uses WARNING log level during feature engineering to avoid logging deadlock on Windows."""

import sys, os, json, pickle, time, random, warnings, logging
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# WARNING level first to avoid logging deadlock during feature_engineering
logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('pipeline')
logger.setLevel(logging.INFO)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
except Exception:
    pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    DATASET_PATH, PREPROCESSED_DIR, RESULTS_DIR, BASELINES_DIR, TABLES_DIR,
    SHAP_DIR, PLOTS_DIR, RECOMMENDATIONS_DIR, N_FOLDS, RANDOM_SEED,
    MODELS_DIR, FT_TRANSFORMER_DIR, SAINT_DIR,
    PAPER_DIR, PAPER_FIGURES_DIR, PAPER_TABLES_DIR,
)

for d in [PREPROCESSED_DIR, BASELINES_DIR, TABLES_DIR, SHAP_DIR, PLOTS_DIR,
          RECOMMENDATIONS_DIR, os.path.join(MODELS_DIR, "tabnet"),
          FT_TRANSFORMER_DIR, SAINT_DIR, PAPER_DIR, PAPER_FIGURES_DIR, PAPER_TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

total_t0 = time.time()

# ===== STAGE 1: PREPROCESS =====
t0 = time.time()
print('=== STAGE 1: PREPROCESS ===')
from src.data.loader import load_dataset
from src.data.preprocessor import clean_data, create_targets, encode_features
from src.data.feature_engineering import engineer_features

df = load_dataset(DATASET_PATH)
df_clean = clean_data(df)
df_eng = engineer_features(df_clean)
y_cgpa, y_at_risk, y_career_ready = create_targets(df_eng)
X, encoders, feature_names = encode_features(df_eng)

np.save(os.path.join(PREPROCESSED_DIR, 'X.npy'), X)
np.save(os.path.join(PREPROCESSED_DIR, 'y_cgpa.npy'), y_cgpa)
np.save(os.path.join(PREPROCESSED_DIR, 'y_at_risk.npy'), y_at_risk)
np.save(os.path.join(PREPROCESSED_DIR, 'y_career_ready.npy'), y_career_ready)
with open(os.path.join(PREPROCESSED_DIR, 'feature_names.json'), 'w') as f:
    json.dump(feature_names, f, indent=2)
with open(os.path.join(PREPROCESSED_DIR, 'encoders.pkl'), 'wb') as f:
    pickle.dump(encoders, f)

print(f'  Preprocessed: X={X.shape}, features={len(feature_names)}, time={time.time()-t0:.1f}s')

# Now enable INFO logging for model training
logging.getLogger().setLevel(logging.INFO)

# ===== STAGE 2: BASELINES =====
t0 = time.time()
print('=== STAGE 2: BASELINES (LR, RF, SVM, XGBoost) ===')
from src.models.baselines import run_all_baselines
baseline_results, baseline_models = run_all_baselines(X, y_cgpa, y_at_risk, y_career_ready, feature_names)
print(f'  Baselines done in {time.time()-t0:.1f}s')

# ===== STAGE 3: TABNET =====
t0 = time.time()
print('=== STAGE 3: TABNET ===')
from src.models.tabnet_model import run_tabnet
tabnet_results, tabnet_models = run_tabnet(X, y_cgpa, y_at_risk, y_career_ready, feature_names)
print(f'  TabNet done in {time.time()-t0:.1f}s')

# ===== STAGE 4: FT-TRANSFORMER =====
t0 = time.time()
print('=== STAGE 4: FT-TRANSFORMER ===')
from src.models.ft_transformer import run_ft_transformer
ft_results = run_ft_transformer(X, y_cgpa, y_at_risk, y_career_ready, feature_names)
print(f'  FT-Transformer done in {time.time()-t0:.1f}s')

# ===== STAGE 5: SAINT =====
t0 = time.time()
print('=== STAGE 5: SAINT ===')
from src.models.saint_model import run_saint
saint_results = run_saint(X, y_cgpa, y_at_risk, y_career_ready, feature_names)
print(f'  SAINT done in {time.time()-t0:.1f}s')

# ===== STAGE 6: SHAP =====
t0 = time.time()
print('=== STAGE 6: SHAP ANALYSIS ===')
from src.explainability.global_shap import run_shap_analysis
run_shap_analysis(X, y_cgpa, y_at_risk, y_career_ready, feature_names)
print(f'  SHAP done in {time.time()-t0:.1f}s')

# ===== STAGE 7: RECOMMENDATIONS =====
t0 = time.time()
print('=== STAGE 7: RECOMMENDATIONS ===')
from src.recommendations.engine import run_recommendations
run_recommendations(X, y_cgpa, y_at_risk, y_career_ready, feature_names, encoders)
print(f'  Recommendations done in {time.time()-t0:.1f}s')

# ===== STAGE 8: VISUALIZATIONS =====
t0 = time.time()
print('=== STAGE 8: VISUALIZATIONS ===')
try:
    import matplotlib
    matplotlib.use('Agg')
    from src.visualization.plots import run_visualizations
    run_visualizations(feature_names)
    print(f'  Visualizations done in {time.time()-t0:.1f}s')
except Exception as e:
    print(f'  Visualizations failed (non-critical): {e}')

# ===== COMBINE DL RESULTS =====
dl_csvs = [
    os.path.join(TABLES_DIR, "tabnet_results.csv"),
    os.path.join(TABLES_DIR, "ft_transformer_results.csv"),
    os.path.join(TABLES_DIR, "saint_results.csv"),
]
frames = [pd.read_csv(p) for p in dl_csvs if os.path.exists(p)]
if frames:
    dl_df = pd.concat(frames, ignore_index=True)
    dl_df.to_csv(os.path.join(TABLES_DIR, "dl_results.csv"), index=False)

print(f'\n=== PIPELINE COMPLETE in {time.time()-total_t0:.1f}s ===')
