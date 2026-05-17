# Table A1. Optuna search spaces per model (TPE sampler, 50 trials, inner 3-fold CV).

| Model | Hyperparameter | Search space |
| --- | --- | --- |
| LogisticRegression | C | [0.01, 100.0] (log) |
| LogisticRegression | solver | {lbfgs, saga} |
| RandomForest | n_estimators | [50, 500] |
| RandomForest | max_depth | [3, 20] |
| RandomForest | min_samples_leaf | [1, 20] |
| SVM | C | [0.01, 100.0] (log) |
| SVM | kernel | {rbf, poly} |
| SVM | gamma | [0.0001, 1.0] (log) |
| XGBoost | n_estimators | [50, 500] |
| XGBoost | max_depth | [3, 10] |
| XGBoost | learning_rate | [0.01, 0.3] (log) |
| XGBoost | subsample | [0.5, 1.0] |
| TabNet | n_d | [8, 64] |
| TabNet | n_steps | [3, 7] |
| TabNet | lambda_sparse | [0.0001, 0.01] (log) |
| TabNet | lr | [0.001, 0.05] (log) |
| FTTransformer | n_blocks | [1, 4] |
| FTTransformer | d_token | [32, 128] |
| FTTransformer | attention_dropout | [0.0, 0.3] |
| FTTransformer | ffn_dropout | [0.0, 0.3] |
| FTTransformer | lr | [0.0001, 0.01] (log) |
| SAINT | depth | [1, 6] |
| SAINT | heads | [2, 8] |
| SAINT | dim | [16, 128] |
| SAINT | attn_dropout | [0.0, 0.3] |
| SAINT | lr | [0.0001, 0.01] (log) |
