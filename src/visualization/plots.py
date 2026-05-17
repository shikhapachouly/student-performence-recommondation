"""Publication-ready visualizations for the student performance prediction pipeline."""

import logging
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    PLOTS_DIR,
    TABLES_DIR,
    SHAP_DIR,
    RECOMMENDATIONS_DIR,
    FEATURE_GROUPS,
)

logger = logging.getLogger(__name__)

# Academic styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.figsize": (10, 6),
})


def save_plot(fig, name, formats=None):
    """Save a figure to results/plots/ in specified formats."""
    if formats is None:
        formats = ["png", "pdf"]
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for fmt in formats:
        path = os.path.join(PLOTS_DIR, f"{name}.{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", name)


def plot_model_comparison(results_df, objective):
    """Grouped bar chart showing F1-macro across all models for an objective."""
    obj_df = results_df[results_df["objective"] == objective]
    if obj_df.empty:
        return

    summary = obj_df.groupby("model_name").agg(
        f1_mean=("f1_macro", "mean"),
        f1_std=("f1_macro", "std"),
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        bal_acc_mean=("balanced_accuracy", "mean"),
        bal_acc_std=("balanced_accuracy", "std"),
    ).reset_index()

    models = summary["model_name"]
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, summary["f1_mean"], width, yerr=summary["f1_std"],
           label="F1-Macro", capsize=3, color="#2196F3")
    ax.bar(x, summary["acc_mean"], width, yerr=summary["acc_std"],
           label="Accuracy", capsize=3, color="#4CAF50")
    ax.bar(x + width, summary["bal_acc_mean"], width, yerr=summary["bal_acc_std"],
           label="Balanced Accuracy", capsize=3, color="#FF9800")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison — {objective.upper()} Prediction")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    save_plot(fig, f"model_comparison_{objective}")


def plot_confusion_matrices(results_df, objective):
    """Heatmap confusion matrices per model for an objective (using last fold)."""
    obj_df = results_df[results_df["objective"] == objective]
    if obj_df.empty:
        return

    models = obj_df["model_name"].unique()
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_df = obj_df[obj_df["model_name"] == model]
        # Use the last fold's confusion matrix
        last_row = model_df.iloc[-1]
        cm = np.array(eval(str(last_row.get("confusion_matrix", "[[0]]"))))
        if cm.ndim < 2:
            continue

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(f"{model}")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    fig.suptitle(f"Confusion Matrices — {objective.upper()}", fontsize=16)
    fig.tight_layout()
    save_plot(fig, f"confusion_matrices_{objective}")


def plot_cv_boxplot(results_df, metric="f1_macro"):
    """Box plot showing metric distribution across folds per model."""
    if results_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, obj in zip(axes, ["cgpa_category", "at_risk", "career_ready"]):
        obj_df = results_df[results_df["objective"] == obj]
        if obj_df.empty:
            continue

        obj_df.boxplot(column=metric, by="model_name", ax=ax, grid=False)
        ax.set_title(f"{obj.upper()} — {metric}")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        plt.sca(ax)
        plt.xticks(rotation=15, ha="right")

    fig.suptitle(f"Cross-Validation {metric} Distribution", fontsize=16)
    fig.tight_layout()
    save_plot(fig, f"cv_boxplot_{metric}")


def plot_shap_summary(feature_names, objective):
    """Bar chart of global SHAP importance from saved CSVs."""
    # Find all global importance files for this objective
    files = glob.glob(os.path.join(SHAP_DIR, f"global_importance_*_{objective}.csv"))
    if not files:
        logger.warning("No SHAP importance files found for %s", objective)
        return

    fig, axes = plt.subplots(1, len(files), figsize=(7 * len(files), 8))
    if len(files) == 1:
        axes = [axes]

    for ax, f in zip(axes, files):
        model_name = os.path.basename(f).replace("global_importance_", "").replace(f"_{objective}.csv", "")
        df = pd.read_csv(f)
        top_n = df.nlargest(15, "mean_abs_shap")

        ax.barh(range(len(top_n)), top_n["mean_abs_shap"].values, color="#2196F3")
        ax.set_yticks(range(len(top_n)))
        ax.set_yticklabels(top_n["feature"].values, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{model_name}")

    fig.suptitle(f"Global Feature Importance (SHAP) — {objective.upper()}", fontsize=16)
    fig.tight_layout()
    save_plot(fig, f"shap_summary_{objective}")


def plot_feature_importance_by_category(feature_names, objective):
    """Grouped bar chart showing SHAP importance by feature category."""
    # Use TabNet SHAP if available, else any model
    imp_path = os.path.join(SHAP_DIR, f"global_importance_TabNet_{objective}.csv")
    if not os.path.exists(imp_path):
        files = glob.glob(os.path.join(SHAP_DIR, f"global_importance_*_{objective}.csv"))
        if files:
            imp_path = files[0]
        else:
            return

    df = pd.read_csv(imp_path)

    # Map features to categories
    feat_to_cat = {}
    for cat, feats in FEATURE_GROUPS.items():
        for feat in feats:
            feat_to_cat[feat] = cat

    df["category"] = df["feature"].map(feat_to_cat).fillna("other")
    cat_importance = df.groupby("category")["mean_abs_shap"].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    cat_importance.plot.barh(ax=ax, color="#4CAF50")
    ax.set_xlabel("Sum of Mean |SHAP value|")
    ax.set_title(f"Feature Importance by Category — {objective.upper()}")
    ax.grid(axis="x", alpha=0.3)

    save_plot(fig, f"feature_importance_by_category_{objective}")


def plot_attention_heatmap(feature_names, objective):
    """Attention importance heatmaps for all DL models that have attention data."""
    att_files = glob.glob(os.path.join(SHAP_DIR, f"attention_importance_*_{objective}.csv"))
    if not att_files:
        return

    n = len(att_files)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8))
    if n == 1:
        axes = [axes]
    colors = ["#FF9800", "#E91E63", "#9C27B0"]

    for ax, att_path, color in zip(axes, att_files, colors * 2):
        model_tag = os.path.basename(att_path).replace("attention_importance_", "").replace(f"_{objective}.csv", "")
        df = pd.read_csv(att_path)
        top_n = df.nlargest(15, "attention_importance")

        ax.barh(range(len(top_n)), top_n["attention_importance"].values, color=color)
        ax.set_yticks(range(len(top_n)))
        ax.set_yticklabels(top_n["feature"].values, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Attention Importance")
        ax.set_title(f"{model_tag}")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"DL Model Attention Importance — {objective.upper()}", fontsize=16)
    fig.tight_layout()
    save_plot(fig, f"attention_heatmap_{objective}")


def plot_recommendation_case_study():
    """Create composite figure showing recommendation case studies."""
    rec_files = glob.glob(os.path.join(RECOMMENDATIONS_DIR, "*.md"))
    if not rec_files:
        return

    # Parse recommendation reports to extract data
    case_data = []
    for rf in rec_files[:4]:  # Max 4 case studies
        with open(rf) as f:
            content = f.read()
        lines = content.split("\n")
        title = lines[0].replace("# ", "") if lines else os.path.basename(rf)
        recs = []
        current_rec = {}
        for line in lines:
            if line.startswith("## Recommendation"):
                if current_rec:
                    recs.append(current_rec)
                current_rec = {"title": line.replace("## ", "")}
            elif line.startswith("- **SHAP Impact**:"):
                try:
                    current_rec["shap"] = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("- **Ranking Score**:"):
                try:
                    current_rec["score"] = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
        if current_rec:
            recs.append(current_rec)
        case_data.append({"title": title, "recs": recs, "file": os.path.basename(rf)})

    if not case_data:
        return

    fig, axes = plt.subplots(1, len(case_data), figsize=(6 * len(case_data), 5))
    if len(case_data) == 1:
        axes = [axes]

    for ax, case in zip(axes, case_data):
        names = [r.get("title", "?")[:20] for r in case["recs"] if "score" in r]
        scores = [r["score"] for r in case["recs"] if "score" in r]
        if names and scores:
            ax.barh(range(len(names)), scores, color="#9C27B0")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Ranking Score")
        ax.set_title(case["file"].replace(".md", ""), fontsize=10)

    fig.suptitle("Recommendation Case Studies", fontsize=14)
    fig.tight_layout()
    save_plot(fig, "recommendation_case_studies")


def run_visualizations(feature_names):
    """Generate all publication-ready visualizations."""
    logger.info("Generating visualizations...")

    # Load all results
    baseline_path = os.path.join(TABLES_DIR, "baseline_results.csv")
    result_csvs = [
        baseline_path,
        os.path.join(TABLES_DIR, "tabnet_results.csv"),
        os.path.join(TABLES_DIR, "ft_transformer_results.csv"),
        os.path.join(TABLES_DIR, "saint_results.csv"),
    ]

    dfs = []
    for csv_path in result_csvs:
        if os.path.exists(csv_path):
            tmp = pd.read_csv(csv_path)
            if "model" in tmp.columns and "model_name" not in tmp.columns:
                tmp = tmp.rename(columns={"model": "model_name"})
            dfs.append(tmp)

    if dfs:
        all_results = pd.concat(dfs, ignore_index=True)
    else:
        logger.warning("No result CSVs found for visualization")
        return

    # Generate comparison table (also saves markdown)
    from src.evaluation.comparison import (
        generate_baseline_comparison,
        generate_model_comparison,
        generate_literature_template,
        generate_literature_report,
    )

    if os.path.exists(baseline_path):
        generate_baseline_comparison(baseline_path)

    if os.path.exists(baseline_path):
        tabnet_path = os.path.join(TABLES_DIR, "tabnet_results.csv")
        generate_model_comparison(baseline_path, tabnet_path if os.path.exists(tabnet_path) else None)

    # Literature template
    generate_literature_template()
    lit_template_path = os.path.join(TABLES_DIR, "literature_template.csv")
    if os.path.exists(lit_template_path):
        generate_literature_report(lit_template_path, proposed_results=True)

    # Plot model comparison for each objective
    for obj in ["cgpa_category", "at_risk", "career_ready"]:
        plot_model_comparison(all_results, obj)

    # Plot CV boxplots
    plot_cv_boxplot(all_results, "f1_macro")
    plot_cv_boxplot(all_results, "accuracy")

    # SHAP visualizations
    for obj in ["cgpa_category", "at_risk", "career_ready"]:
        plot_shap_summary(feature_names, obj)
        plot_feature_importance_by_category(feature_names, obj)
        plot_attention_heatmap(feature_names, obj)

    # Recommendation case studies
    plot_recommendation_case_study()

    # Count generated plots
    # Confusion matrices
    for obj in ["cgpa_category", "at_risk", "career_ready"]:
        plot_confusion_matrices(all_results, obj)

    plot_files = glob.glob(os.path.join(PLOTS_DIR, "*.*"))
    logger.info("Generated %d plot files in %s (SC-007 target: >= 15)", len(plot_files), PLOTS_DIR)
