"""Model comparison tables and literature comparison framework."""

import logging
import os

import numpy as np
import pandas as pd

from src.config import TABLES_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)


def generate_baseline_comparison(results_csv_path):
    """Read baseline_results.csv and produce a formatted summary table (model x objective x metric)."""
    df = pd.read_csv(results_csv_path)

    summary_rows = []
    for (model, obj), group in df.groupby(["model_name", "objective"]):
        row = {
            "Model": model,
            "Objective": obj,
            "Accuracy": f"{group['accuracy'].mean():.4f} (+/- {group['accuracy'].std():.4f})",
            "F1-Macro": f"{group['f1_macro'].mean():.4f} (+/- {group['f1_macro'].std():.4f})",
            "F1-Weighted": f"{group['f1_weighted'].mean():.4f} (+/- {group['f1_weighted'].std():.4f})",
            "Balanced Acc": f"{group['balanced_accuracy'].mean():.4f} (+/- {group['balanced_accuracy'].std():.4f})",
        }
        if group["auc_roc"].notna().any():
            auc_vals = group["auc_roc"].dropna()
            row["AUC-ROC"] = f"{auc_vals.mean():.4f} (+/- {auc_vals.std():.4f})"
        else:
            row["AUC-ROC"] = "N/A"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Save as markdown
    md_path = os.path.join(TABLES_DIR, "baseline_summary.md")
    with open(md_path, "w") as f:
        f.write("# Baseline Model Results Summary\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n")

    logger.info("Baseline summary saved to %s", md_path)
    return summary_df


def generate_model_comparison(baseline_csv, tabnet_csv=None):
    """Produce side-by-side comparison of all models with improvement percentages."""
    frames = [pd.read_csv(baseline_csv)]

    # Dynamically load all available DL result CSVs (individual per model)
    dl_csvs = [
        tabnet_csv,
        os.path.join(TABLES_DIR, "ft_transformer_results.csv"),
        os.path.join(TABLES_DIR, "saint_results.csv"),
    ]
    loaded_models = set()
    for csv_path in dl_csvs:
        if csv_path and os.path.exists(csv_path):
            tmp = pd.read_csv(csv_path)
            # Normalize column names
            if "model" in tmp.columns and "model_name" not in tmp.columns:
                tmp = tmp.rename(columns={"model": "model_name"})
            if "model_name" not in tmp.columns:
                logger.warning("Skipping %s: no model_name column found", csv_path)
                continue
            # Avoid duplicates
            for m in tmp["model_name"].unique():
                if m not in loaded_models:
                    frames.append(tmp[tmp["model_name"] == m])
                    loaded_models.add(m)

    metrics = ["accuracy", "f1_macro", "f1_weighted", "balanced_accuracy"]
    comparison_rows = []

    for frame in frames:
        is_fold_level = "fold" in frame.columns

        if is_fold_level:
            # Fold-level data (baselines, TabNet): aggregate across folds
            for obj in frame["objective"].unique():
                obj_data = frame[frame["objective"] == obj]
                for model in obj_data["model_name"].unique():
                    model_data = obj_data[obj_data["model_name"] == model]
                    row = {"model_name": model, "objective": obj}
                    for metric in metrics:
                        row[f"{metric}_mean"] = model_data[metric].mean()
                        row[f"{metric}_std"] = model_data[metric].std()
                    if "auc_roc" in model_data.columns and model_data["auc_roc"].notna().any():
                        row["auc_roc_mean"] = model_data["auc_roc"].dropna().mean()
                        row["auc_roc_std"] = model_data["auc_roc"].dropna().std()
                    comparison_rows.append(row)
        else:
            # Pre-aggregated data (FT-Transformer, SAINT): already has _mean/_std columns
            for _, dl_row in frame.iterrows():
                row = {"model_name": dl_row["model_name"], "objective": dl_row["objective"]}
                for metric in metrics:
                    row[f"{metric}_mean"] = dl_row.get(f"{metric}_mean", None)
                    row[f"{metric}_std"] = dl_row.get(f"{metric}_std", None)
                if "auc_roc_mean" in dl_row.index:
                    row["auc_roc_mean"] = dl_row.get("auc_roc_mean", None)
                    row["auc_roc_std"] = dl_row.get("auc_roc_std", None)
                comparison_rows.append(row)

    comp_df = pd.DataFrame(comparison_rows)

    # Compute improvement of each DL model vs best baseline per objective
    from src.config import BASELINE_MODELS
    sc001_achieved = False
    for obj in comp_df["objective"].unique():
        obj_df = comp_df[comp_df["objective"] == obj]
        baselines = obj_df[obj_df["model_name"].isin(BASELINE_MODELS)]
        dl_models = obj_df[~obj_df["model_name"].isin(BASELINE_MODELS)]

        if baselines.empty or dl_models.empty:
            continue

        best_baseline_f1 = baselines["f1_macro_mean"].max()
        best_baseline_name = baselines.loc[baselines["f1_macro_mean"].idxmax(), "model_name"]

        for _, dl_row in dl_models.iterrows():
            dl_name = dl_row["model_name"]
            dl_f1 = dl_row["f1_macro_mean"]

            if best_baseline_f1 > 0:
                relative_improvement = ((dl_f1 - best_baseline_f1) / best_baseline_f1) * 100
            else:
                relative_improvement = 0

            abs_improvement = dl_f1 - best_baseline_f1

            logger.info(
                "SC-001 check (%s): %s F1-macro=%.4f vs best baseline (%s)=%.4f | "
                "Improvement: %.2f%% (absolute: %+.4f)",
                obj, dl_name, dl_f1, best_baseline_name, best_baseline_f1,
                relative_improvement, abs_improvement,
            )

            if relative_improvement >= 5.0:
                sc001_achieved = True
                logger.info("  SC-001 ACHIEVED for %s with %s (>= 5%% relative improvement)", obj, dl_name)

    if not sc001_achieved:
        logger.warning("SC-001 NOT achieved on any objective (< 5%% relative improvement)")

    # Save comparison
    comp_csv_path = os.path.join(TABLES_DIR, "comparison_summary.csv")
    comp_df.to_csv(comp_csv_path, index=False)

    # Save as markdown
    comp_md_path = os.path.join(TABLES_DIR, "comparison_summary.md")
    with open(comp_md_path, "w") as f:
        f.write("# Model Comparison Summary\n\n")
        for obj in comp_df["objective"].unique():
            f.write(f"\n## Objective: {obj}\n\n")
            obj_df = comp_df[comp_df["objective"] == obj][
                ["model_name", "f1_macro_mean", "f1_macro_std", "accuracy_mean",
                 "balanced_accuracy_mean"]
            ].copy()
            obj_df.columns = ["Model", "F1-Macro (mean)", "F1-Macro (std)",
                              "Accuracy (mean)", "Balanced Acc (mean)"]
            f.write(obj_df.to_markdown(index=False))
            f.write("\n")

    logger.info("Comparison summary saved to %s", comp_csv_path)
    return comp_df


def generate_literature_template():
    """Create a literature comparison template CSV and Markdown report."""
    template_data = {
        "Study": ["Proposed (This Work)"],
        "Year": [2026],
        "Dataset_Size": [1378],
        "Dataset_Features": [48],
        "Method": ["TabNet + FT-Transformer + SAINT + SHAP"],
        "Accuracy": [""],
        "F1_Macro": [""],
        "AUC_ROC": [""],
        "RMSE": ["N/A"],
        "Notes": ["Multi-objective: CGPA Category + At-Risk + Career Readiness"],
    }

    template_df = pd.DataFrame(template_data)
    template_path = os.path.join(TABLES_DIR, "literature_template.csv")
    template_df.to_csv(template_path, index=False)

    # Generate markdown report template
    report_path = os.path.join(REPORTS_DIR, "literature_comparison.md")
    with open(report_path, "w") as f:
        f.write("# Literature Comparison Report\n\n")
        f.write("## Instructions\n\n")
        f.write("Fill in the `literature_template.csv` with results from published studies.\n")
        f.write("Then re-run the pipeline to generate the comparison narrative.\n\n")
        f.write("## Comparison Table\n\n")
        f.write(template_df.to_markdown(index=False))
        f.write("\n\n## Analysis\n\n")
        f.write("*Auto-generated after template is filled.*\n")

    logger.info("Literature template saved to %s", template_path)
    return template_df


def generate_literature_report(template_path, proposed_results=None):
    """Read the filled template and generate a comparison report with narratives."""
    df = pd.read_csv(template_path)

    # Auto-fill proposed model results if available
    if proposed_results is not None:
        comp_csv = os.path.join(TABLES_DIR, "comparison_summary.csv")
        if os.path.exists(comp_csv):
            comp_df = pd.read_csv(comp_csv)
            # Convert numeric columns to string for safe assignment
            for col in ["Accuracy", "F1_Macro", "AUC_ROC", "RMSE"]:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            tabnet_rows = comp_df[comp_df["model_name"] == "TabNet"]
            for _, row in tabnet_rows.iterrows():
                mask = df["Study"] == "Proposed (This Work)"
                if mask.any():
                    df.loc[mask, "Accuracy"] = f"{row.get('accuracy_mean', ''):.4f}" if pd.notna(row.get('accuracy_mean')) else ""
                    df.loc[mask, "F1_Macro"] = f"{row.get('f1_macro_mean', ''):.4f}" if pd.notna(row.get('f1_macro_mean')) else ""
                    if pd.notna(row.get("auc_roc_mean")):
                        df.loc[mask, "AUC_ROC"] = f"{row.get('auc_roc_mean', ''):.4f}"

    report_path = os.path.join(REPORTS_DIR, "literature_comparison.md")
    with open(report_path, "w") as f:
        f.write("# Literature Comparison Report\n\n")
        f.write("## Comparison Table\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Analysis\n\n")

        # Generate narrative for each study
        proposed = df[df["Study"] == "Proposed (This Work)"]
        others = df[df["Study"] != "Proposed (This Work)"]

        if not proposed.empty and not others.empty:
            f.write("### Comparison with Published Studies\n\n")
            for _, study in others.iterrows():
                f.write(f"**{study['Study']} ({study['Year']})**: ")
                f.write(f"Used {study['Method']} on a dataset of {study['Dataset_Size']} samples. ")
                if pd.notna(study.get("Notes")):
                    f.write(f"{study['Notes']}. ")
                f.write("\n\n")
        else:
            f.write("*Fill in the literature_template.csv with published results to generate comparison.*\n")

    logger.info("Literature report updated at %s", report_path)
    return df
