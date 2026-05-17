"""FT-Transformer training with cross-validation and SMOTE."""

import logging
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

from src.config import (
    FT_BATCH_SIZE,
    FT_D_TOKEN,
    FT_LR,
    FT_MAX_EPOCHS,
    FT_N_BLOCKS,
    FT_PATIENCE,
    FT_TRANSFORMER_DIR,
    N_FOLDS,
    RANDOM_SEED,
    TABLES_DIR,
)
from src.evaluation.metrics import compute_metrics, aggregate_fold_results

logger = logging.getLogger(__name__)


class SimpleFTTransformer(nn.Module):
    """Simplified FT-Transformer for tabular data (pure PyTorch)."""

    def __init__(self, n_features, n_classes, d_token=64, n_blocks=2,
                 n_heads=4, attn_dropout=0.2, ffn_dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token

        # Feature tokenizer: project each feature to d_token
        self.feature_embeddings = nn.Linear(1, d_token)
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
            dropout=ffn_dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, n_classes),
        )

    def forward(self, x):
        # x: (batch, n_features)
        batch_size = x.size(0)
        # Tokenize each feature independently
        tokens = self.feature_embeddings(x.unsqueeze(-1))  # (batch, n_features, d_token)
        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (batch, n_features+1, d_token)
        # Transformer
        out = self.transformer(tokens)
        out = self.attn_dropout(out)
        # Use CLS token output for classification
        cls_out = out[:, 0, :]
        return self.head(cls_out)

    def get_attention_weights(self, x):
        """Extract attention weights from the last transformer layer."""
        self.eval()
        with torch.no_grad():
            batch_size = x.size(0)
            tokens = self.feature_embeddings(x.unsqueeze(-1))
            cls = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            # Get attention from last layer
            last_layer = self.transformer.layers[-1]
            # Forward through all layers except last
            out = tokens
            for layer in self.transformer.layers[:-1]:
                out = layer(out)
            # Get attention from last layer's self_attn
            attn_out, attn_weights = last_layer.self_attn(
                out, out, out, need_weights=True, average_attn_weights=True
            )
        # attn_weights: (batch, seq_len, seq_len)
        # Extract CLS attention to features (skip CLS-to-CLS)
        feature_attn = attn_weights[:, 0, 1:]  # (batch, n_features)
        return feature_attn.numpy()


def train_ft_transformer_cv(X, y, objective, params=None, n_folds=N_FOLDS, seed=RANDOM_SEED):
    """Train FT-Transformer with StratifiedKFold CV and SMOTE.

    Returns:
        fold_results: List of per-fold metric dicts.
        attention_importance: Mean attention per feature across folds.
        fold_models: List of trained models.
    """
    if params is None:
        params = {}

    d_token = params.get("d_token", FT_D_TOKEN)
    n_blocks = params.get("n_blocks", FT_N_BLOCKS)
    attn_dropout = params.get("attention_dropout", 0.2)
    ffn_dropout = params.get("ffn_dropout", 0.1)
    lr = params.get("lr", FT_LR)
    max_epochs = params.get("max_epochs", FT_MAX_EPOCHS)
    patience = params.get("patience", FT_PATIENCE)
    batch_size = params.get("batch_size", FT_BATCH_SIZE)

    n_classes = len(np.unique(y))
    n_features = X.shape[1]

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []
    fold_models = []
    all_attn = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        logger.info("  FTTransformer | %s | Fold %d/%d", objective, fold_idx + 1, n_folds)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train/valid split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=seed
        )

        # SMOTE
        sm = SMOTE(random_state=seed)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

        # Class weights for loss
        cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        weight_tensor = torch.FloatTensor(cw)

        # Build model
        model = SimpleFTTransformer(
            n_features=n_features, n_classes=n_classes,
            d_token=d_token, n_blocks=n_blocks,
            attn_dropout=attn_dropout, ffn_dropout=ffn_dropout,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        # Convert to tensors
        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_t = torch.LongTensor(y_tr)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)
        X_test_t = torch.FloatTensor(X_test)

        # Training loop
        best_val_loss = float("inf")
        wait = 0
        best_state = None

        for epoch in range(max_epochs):
            model.train()
            # Mini-batch training
            perm = torch.randperm(len(X_tr_t))
            epoch_loss = 0
            n_batches = 0
            for i in range(0, len(X_tr_t), batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = X_tr_t[idx], y_tr_t[idx]
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(X_val_t)
                val_loss = criterion(val_out, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        # Load best
        if best_state:
            model.load_state_dict(best_state)

        # Predict
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t)
            y_pred = logits.argmax(dim=1).numpy()
            y_proba = torch.softmax(logits, dim=1).numpy()

        # Metrics
        metrics = compute_metrics(y_test, y_pred, y_proba, objective)
        fold_results.append(metrics)
        fold_models.append(model)

        # Attention weights
        attn = model.get_attention_weights(X_test_t)
        all_attn.append(attn.mean(axis=0))

        logger.info("    Fold %d: accuracy=%.4f, f1_macro=%.4f, balanced_acc=%.4f",
                     fold_idx, metrics["accuracy"], metrics["f1_macro"], metrics["balanced_accuracy"])

    # Aggregate attention importance
    attention_importance = np.mean(all_attn, axis=0) if all_attn else np.zeros(n_features)

    # Aggregate results
    agg = aggregate_fold_results(fold_results)
    logger.info("  FTTransformer | %s | Mean F1-macro: %.4f (+/- %.4f)",
                objective, agg["f1_macro_mean"], agg["f1_macro_std"])

    return fold_results, attention_importance, fold_models


def run_ft_transformer(X, y_cgpa, y_at_risk, y_career_ready, feature_names, params=None):
    """Train FT-Transformer for both objectives."""
    import pandas as pd
    os.makedirs(FT_TRANSFORMER_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    all_results = []
    objectives = {"cgpa_category": y_cgpa, "at_risk": y_at_risk, "career_ready": y_career_ready}

    for obj_name, y in objectives.items():
        logger.info("Training FT-Transformer for %s...", obj_name)
        obj_params = params.get(obj_name, {}).get("FTTransformer", {}) if params else {}
        fold_results, attn_imp, fold_models = train_ft_transformer_cv(
            X, y, obj_name, params=obj_params
        )

        # Save attention importance
        import pandas as pd
        attn_df = pd.DataFrame({
            "feature": feature_names,
            "attention_importance": attn_imp,
        }).sort_values("attention_importance", ascending=False)
        from src.config import SHAP_DIR
        os.makedirs(SHAP_DIR, exist_ok=True)
        attn_df.to_csv(os.path.join(SHAP_DIR, f"attention_importance_fttransformer_{obj_name}.csv"), index=False)

        # Save best model
        best_idx = max(range(len(fold_results)), key=lambda i: fold_results[i]["f1_macro"])
        model_path = os.path.join(FT_TRANSFORMER_DIR, f"fttransformer_{obj_name}_best.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(fold_models[best_idx], f)

        agg = aggregate_fold_results(fold_results)
        agg["model"] = "FTTransformer"
        agg["objective"] = obj_name
        all_results.append(agg)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(TABLES_DIR, "ft_transformer_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info("FT-Transformer results saved to %s", results_path)
    return all_results
