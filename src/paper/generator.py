"""Research paper draft generation from pipeline results."""

import logging
import os
import glob
import shutil

from src.config import PLOTS_DIR, PAPER_DIR, PAPER_FIGURES_DIR, PAPER_TABLES_DIR, TABLES_DIR
from src.paper.sections import (
    generate_abstract,
    generate_introduction,
    generate_methodology,
    generate_experimental_setup,
    generate_results,
    generate_conclusion,
)

logger = logging.getLogger(__name__)


def generate_paper(results_dir, output_dir):
    """Generate the full research paper draft from pipeline results.

    Args:
        results_dir: Path to the results directory.
        output_dir: Path to the paper output directory.
    """
    logger.info("Generating research paper draft...")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(PAPER_FIGURES_DIR, exist_ok=True)
    os.makedirs(PAPER_TABLES_DIR, exist_ok=True)

    # Load template
    template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "templates", "paper_template.md"
    )

    with open(template_path) as f:
        template = f.read()

    # Generate each section
    title = (
        "Multi-Objective Student Performance Prediction Using Explainable "
        "Artificial Intelligence: A TabNet-SHAP Framework with Personalized "
        "Recommendation System"
    )
    abstract = generate_abstract(results_dir)
    introduction = generate_introduction()
    methodology = generate_methodology()
    experimental_setup = generate_experimental_setup(results_dir)
    results_section = generate_results(results_dir)
    conclusion = generate_conclusion(results_dir)

    literature_review = """### 2.1 Student Performance Prediction

*[Researcher: Add literature review entries from your thesis literature survey here.]*

| Study | Year | Dataset | Method | Key Findings |
|-------|------|---------|--------|-------------|
| *[Add entry]* | | | | |
| *[Add entry]* | | | | |
| *[Add entry]* | | | | |

### 2.2 Explainable AI in Education

*[Researcher: Add XAI-specific literature here.]*

### 2.3 Recommendation Systems for Education

*[Researcher: Add recommendation system literature here.]*

### 2.4 Research Gap

The existing literature reveals a gap in combining multi-objective prediction with 
explainability and actionable recommendations in a unified framework. Most studies 
address prediction accuracy alone, without translating insights into student-facing 
interventions. This study addresses this gap through an integrated TabNet-SHAP-Recommendation 
pipeline."""

    references = """*[Researcher: Add your references in the appropriate citation format for the target journal.]*

1. Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. *AAAI*.
2. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
3. *[Add additional references]*"""

    # Replace placeholders
    paper = template.replace("{{TITLE}}", title)
    paper = paper.replace("{{ABSTRACT}}", abstract)
    paper = paper.replace("{{INTRODUCTION}}", introduction)
    paper = paper.replace("{{LITERATURE_REVIEW}}", literature_review)
    paper = paper.replace("{{METHODOLOGY}}", methodology)
    paper = paper.replace("{{EXPERIMENTAL_SETUP}}", experimental_setup)
    paper = paper.replace("{{RESULTS}}", results_section)
    paper = paper.replace("{{CONCLUSION}}", conclusion)
    paper = paper.replace("{{REFERENCES}}", references)

    # Check for remaining placeholders
    remaining = [p for p in ["{{"] if p in paper]
    if remaining:
        logger.warning("Paper still contains unresolved placeholders")

    # Save paper draft
    paper_path = os.path.join(output_dir, "paper_draft.md")
    with open(paper_path, "w", encoding="utf-8") as f:
        f.write(paper)
    logger.info("Paper draft saved to %s", paper_path)

    # Copy referenced plot files to paper/figures/
    plot_files = glob.glob(os.path.join(PLOTS_DIR, "*.png"))
    for pf in plot_files:
        dest = os.path.join(PAPER_FIGURES_DIR, os.path.basename(pf))
        shutil.copy2(pf, dest)
    logger.info("Copied %d figures to %s", len(plot_files), PAPER_FIGURES_DIR)

    # Copy table files to paper/tables/
    table_files = glob.glob(os.path.join(TABLES_DIR, "*.md")) + glob.glob(os.path.join(TABLES_DIR, "*.csv"))
    for tf in table_files:
        dest = os.path.join(PAPER_TABLES_DIR, os.path.basename(tf))
        shutil.copy2(tf, dest)
    logger.info("Copied %d tables to %s", len(table_files), PAPER_TABLES_DIR)

    # Count figure/table references in paper
    fig_refs = paper.count("![")
    table_refs = paper.count("| ")
    logger.info(
        "Paper contains %d figure references and %d table rows (SC-008 target: >= 10 total)",
        fig_refs, table_refs,
    )

    # Verify no remaining template placeholders
    if "{{" in paper:
        unresolved = [line for line in paper.split("\n") if "{{" in line]
        logger.warning("Unresolved placeholders: %s", unresolved)
    else:
        logger.info("All template placeholders resolved successfully")

    return paper_path
