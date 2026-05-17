"""SAINT (Self-Attention and Intersample Attention) model training."""

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
    N_FOLDS,
    RANDOM_SEED,
    SAINT_BATCH_SIZE,
    SAINT_DEPTH,
    SAINT_DIM,
    SAINT_DIR,
    SAINT_FF_DROPOUT,
    SAINT_HEADS,
    SAINT_LR,
    SAINT_MAX_EPOCHS,
    SAINT_PATIENCE,
    TABLES_DIR,
)
from src.evaluation.metrics import compute_metrics, aggregate_fold_results

logger = logging.getLogger(__name__)


class SimpleSAINT(nn.Module):
    """Simplified SAINT: self-attention across features + intersample attention."""

    def __init__(self, n_features, n_classes, dim=32, depth=3, heads=4,
                 attn_dropout=0.1, ff_dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.dim = dim

        # Feature embedding
        self.feature_embed = nn.Linear(1, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Intra-sample self-attention layers (across features within a sample)
        self.intra_layers = nn.ModuleList()
        for _ in range(depth):
            self.intra_layers.append(nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 4,
                dropout=ff_dropout, batch_first=True,
            ))

        # Inter-sample attention layer (across samples — simplified as MLP mixer)
        self.inter_norm = nn.LayerNorm(dim)
        self.inter_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * 2, dim),
        )

        self.attn_dropout = nn.Dropout(attn_dropout)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        # Tokenize features
        tokens = self.feature_embed(x.unsqueeze(-1))  # (batch, n_features, dim)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (batch, n_features+1, dim)

        # Intra-sample attention
        for layer in self.intra_layers:
            tokens = layer(tokens)

        tokens = self.attn_dropout(tokens)

        # Inter-sample: apply MLP across the batch dimension on CLS tokens
        cls_out = tokens[:, 0, :]  # (batch, dim)
        inter_out = self.inter_mlp(self.inter_norm(cls_out)) + cls_out

        return self.head(inter_out)

    def get_intra_attention(self, x):
        """Extract intra-sample attention from last layer."""
        self.eval()
        with torch.no_grad():
            batch_size = x.size(0)
            tokens = self.feature_embed(x.unsqueeze(-1))
            cls = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

            # Forward through all layers except last
            for layer in self.intra_layers[:-1]:
                tokens = layer(tokens)

            # Get attention from last layer
            last_layer = self.intra_layers[-1]
            _, attn_weights = last_layer.self_attn(
                tokens, tokens, tokens, need_weights=True, average_attn_weights=True
            )
        # CLS attention to features
        feature_attn = attn_weights[:, 0, 1:]  # (batch, n_features)
        return feature_attn.numpy()


def train_saint_cv(X, y, objective, params=None, n_folds=N_FOLDS, seed=RANDOM_SEED):
    """Train SAINT with StratifiedKFold CV and SMOTE."""
    if params is None:
        params = {}

    dim = params.get("dim", SAINT_DIM)
    depth = params.get("depth", SAINT_DEPTH)
    heads = params.get("heads", SAINT_HEADS)
    attn_dropout = params.get("attn_dropout", 0.1)
    ff_dropout = params.get("ff_dropout", SAINT_FF_DROPOUT)
    lr = params.get("lr", SAINT_LR)
    max_epochs = params.get("max_epochs", SAINT_MAX_EPOCHS)
    patience = params.get("patience", SAINT_PATIENCE)
    batch_size = params.get("batch_size", SAINT_BATCH_SIZE)

    n_classes = len(np.unique(y))
    n_features = X.shape[1]

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []
    fold_models = []
    all_attn = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        logger.info("  SAINT | %s | Fold %d/%d", objective, fold_idx + 1, n_folds)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=seed
        )

        sm = SMOTE(random_state=seed)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

        cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        weight_tensor = torch.FloatTensor(cw)

        model = SimpleSAINT(
            n_features=n_features, n_classes=n_classes,
            dim=dim, depth=depth, heads=heads,
            attn_dropout=attn_dropout, ff_dropout=ff_dropout,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_t = torch.LongTensor(y_tr)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)
        X_test_t = torch.FloatTensor(X_test)

        best_val_loss = float("inf")
        wait = 0
        best_state = None

        for epoch in range(max_epochs):
            model.train()
            perm = torch.randperm(len(X_tr_t))
            for i in range(0, len(X_tr_t), batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = X_tr_t[idx], y_tr_t[idx]
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

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

        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            logits = model(X_test_t)
            y_pred = logits.argmax(dim=1).numpy()
            y_proba = torch.softmax(logits, dim=1).numpy()

        metrics = compute_metrics(y_test, y_pred, y_proba, objective)
        fold_results.append(metrics)
        fold_models.append(model)

        attn = model.get_intra_attention(X_test_t)
        all_attn.append(attn.mean(axis=0))

        logger.info("    Fold %d: accuracy=%.4f, f1_macro=%.4f, balanced_acc=%.4f",
                     fold_idx, metrics["accuracy"], metrics["f1_macro"], metrics["balanced_accuracy"])

    attention_importance = np.mean(all_attn, axis=0) if all_attn else np.zeros(n_features)

    agg = aggregate_fold_results(fold_results)
    logger.info("  SAINT | %s | Mean F1-macro: %.4f (+/- %.4f)",
                objective, agg["f1_macro_mean"], agg["f1_macro_std"])

    return fold_results, attention_importance, fold_models


def run_saint(X, y_cgpa, y_at_risk, y_career_ready, feature_names, params=None):
    """Train SAINT for both objectives."""
    import pandas as pd
    os.makedirs(SAINT_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    all_results = []
    objectives = {"cgpa_category": y_cgpa, "at_risk": y_at_risk, "career_ready": y_career_ready}

    for obj_name, y in objectives.items():
        logger.info("Training SAINT for %s...", obj_name)
        obj_params = params.get(obj_name, {}).get("SAINT", {}) if params else {}
        fold_results, attn_imp, fold_models = train_saint_cv(
            X, y, obj_name, params=obj_params
        )

        # Save attention importance
        attn_df = pd.DataFrame({
            "feature": feature_names,
            "attention_importance": attn_imp,
        }).sort_values("attention_importance", ascending=False)
        from src.config import SHAP_DIR
        os.makedirs(SHAP_DIR, exist_ok=True)
        attn_df.to_csv(os.path.join(SHAP_DIR, f"attention_importance_saint_{obj_name}.csv"), index=False)

        best_idx = max(range(len(fold_results)), key=lambda i: fold_results[i]["f1_macro"])
        model_path = os.path.join(SAINT_DIR, f"saint_{obj_name}_best.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(fold_models[best_idx], f)

        agg = aggregate_fold_results(fold_results)
        agg["model"] = "SAINT"
        agg["objective"] = obj_name
        all_results.append(agg)

    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(TABLES_DIR, "saint_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info("SAINT results saved to %s", results_path)
    return all_results
