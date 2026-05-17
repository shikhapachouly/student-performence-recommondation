import os

# === Paths ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "student-dataset.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PREPROCESSED_DIR = os.path.join(RESULTS_DIR, "preprocessed")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
BASELINES_DIR = os.path.join(MODELS_DIR, "baselines")
TABNET_DIR = os.path.join(MODELS_DIR, "tabnet")
FT_TRANSFORMER_DIR = os.path.join(MODELS_DIR, "ft_transformer")
SAINT_DIR = os.path.join(MODELS_DIR, "saint")
SHAP_DIR = os.path.join(RESULTS_DIR, "shap")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
RECOMMENDATIONS_DIR = os.path.join(REPORTS_DIR, "recommendations")
PAPER_DIR = os.path.join(RESULTS_DIR, "paper")
PAPER_FIGURES_DIR = os.path.join(PAPER_DIR, "figures")
PAPER_TABLES_DIR = os.path.join(PAPER_DIR, "tables")

# === Reproducibility ===
RANDOM_SEED = 42
N_FOLDS = 5

# === CGPA Categorization Thresholds ===
CGPA_LOW_THRESHOLD = 6.5
CGPA_HIGH_THRESHOLD = 8.0

# === Suspicious Imputed Values (replace with NaN) ===
SUSPICIOUS_CGPA = 7.726610922
SUSPICIOUS_12TH = 75.19833333

# === TabNet Hyperparameters ===
TABNET_N_D = 8
TABNET_N_A = 8
TABNET_N_STEPS = 3
TABNET_LAMBDA_SPARSE = 1e-3
TABNET_MAX_EPOCHS = 300
TABNET_PATIENCE = 20
TABNET_BATCH_SIZE = 128
TABNET_LR = 2e-2
TABNET_GAMMA = 1.3
TABNET_MOMENTUM = 0.02
TABNET_CLIP_VALUE = 1.0
TABNET_SCHEDULER_STEP = 50
TABNET_SCHEDULER_GAMMA = 0.9
TABNET_MASK_TYPE = "sparsemax"
TABNET_VALID_SPLIT = 0.15

# === SHAP Configuration ===
SHAP_BACKGROUND_SAMPLES = 100
SHAP_NSAMPLES = 1000

# === Features to Drop ===
DROP_FEATURES = ["Timestamp", "Username", "NameOfStudent", "hobbies"]

# === Target Features (raw columns used to derive targets — dropped from feature set to prevent data leakage) ===
TARGET_FEATURES = [
    "current_cgpa",                     # used for cgpa_category target
    "backlog_status",                   # used for at_risk target
    "internship_completion",            # used for career_ready target
    "academic_projects_completed_yn",   # used for career_ready target
    "internship_payment_status",        # proxy for internship_completion — dropped to prevent career_ready leakage
    "placement_status",                 # outcome variable — not predicted, not a feature
    "higher_education_interest_yn",     # outcome variable — not predicted, not a feature
]

# === Leakage Safeguards ===
# Engineered features are also checked for leakage:
#   - academic_index: computed from (tenth_percentage, twelth_percentage) ONLY.
#     current_cgpa is EXCLUDED because it defines cgpa_category and at_risk targets.
#   - engagement_score: computed from (extracurricular_support, workshop_seminar_participation_yn,
#     community_service_yn) ONLY. internship_completion and academic_projects_completed_yn are
#     EXCLUDED because they define the career_ready target.
#   - skills_index, study_stress_interaction, skills_study_interaction: no target overlap.
#
# Explicit predictor lists per task (all share the same feature matrix X after target column removal):
#   Common features after dropping TARGET_FEATURES and LOW_VARIANCE_DROP:
#     - Demographic: gender, residence_type, institute_name
#     - Prior academic: tenth_percentage, twelth_percentage, course_branch, nursery_education
#     - Family: mother_education_level, mother_occupation, father_education_level,
#       father_occupation, current_guardian, family_size, parental_marital_status,
#       family_relationship_quality_scale_1to5, family_income_bracket
#     - Behavioral: daily_study_hours, extracurricular_support,
#       workshop_seminar_participation_yn, community_service_yn, commute_transport_mode,
#       commute_duration, socializing_frequency_scale_1to5, romantic_relationship_yn,
#       paid_classes_attending_yn
#     - Skills: communication_skills_scale_1to5, leadership_scale_1to5,
#       teamwork_skills_scale_1to5, peer_group_quality_scale_1to5, time_management_scale_1to5
#     - Wellbeing: health_status_scale_1to5, stress_frequency_scale_1to5, stress_coping_yn
#     - Context: college_choice_reason, course_choice_reason, internet_access,
#       family_extra_classes_motivation_status
#     - Engineered: academic_index (10th+12th only), skills_index, engagement_score,
#       study_stress_interaction, skills_study_interaction

# === Feature Group Mappings (per data-model.md) ===
FEATURE_GROUPS = {
    "academic": [
        "tenth_percentage", "twelth_percentage", "current_cgpa",
        "backlog_status", "course_branch", "nursery_education"
    ],
    "demographic": [
        "gender", "age_group", "residence_type", "institute_name"
    ],
    "family": [
        "mother_education_level", "mother_occupation",
        "father_education_level", "father_occupation",
        "current_guardian", "family_size", "parental_marital_status",
        "family_relationship_quality_scale_1to5", "family_income_bracket"
    ],
    "behavioral": [
        "daily_study_hours", "extracurricular_support",
        "workshop_seminar_participation_yn", "community_service_yn",
        "commute_transport_mode", "commute_duration",
        "socializing_frequency_scale_1to5", "romantic_relationship_yn",
        "alcohol_use", "tobacco_use", "other_addictions",
        "paid_classes_attending_yn"
    ],
    "skills": [
        "communication_skills_scale_1to5", "leadership_scale_1to5",
        "teamwork_skills_scale_1to5", "internship_completion",
        "internship_payment_status", "academic_projects_completed_yn",
        "peer_group_quality_scale_1to5", "time_management_scale_1to5"
    ],
    "wellbeing": [
        "health_status_scale_1to5", "stress_frequency_scale_1to5",
        "stress_coping_yn"
    ],
    "context": [
        "college_choice_reason", "course_choice_reason",
        "internet_access", "family_extra_classes_motivation_status"
    ],
}

# === Modifiable Features (for recommendations) ===
MODIFIABLE_FEATURES = {
    "daily_study_hours": {
        "display_name": "Daily Study Hours",
        "category": "behavioral",
        "direction": "increase",
        "ordinal_order": ["15 to 30 min", "30 min to 1 hour", "1 hour - 4 hour", "More than 4 hours"],
    },
    "extracurricular_support": {
        "display_name": "Extracurricular Support",
        "category": "behavioral",
        "direction": "activate",
        "ordinal_order": [0, 1],
    },
    "workshop_seminar_participation_yn": {
        "display_name": "Workshop/Seminar Participation",
        "category": "behavioral",
        "direction": "activate",
        "ordinal_order": [0, 1],
    },
    "community_service_yn": {
        "display_name": "Community Service",
        "category": "behavioral",
        "direction": "activate",
        "ordinal_order": [0, 1],
    },
    "communication_skills_scale_1to5": {
        "display_name": "Communication Skills",
        "category": "skills",
        "direction": "increase",
        "ordinal_order": [1, 2, 3, 4, 5],
    },
    "leadership_scale_1to5": {
        "display_name": "Leadership Skills",
        "category": "skills",
        "direction": "increase",
        "ordinal_order": [1, 2, 3, 4, 5],
    },
    "teamwork_skills_scale_1to5": {
        "display_name": "Teamwork Skills",
        "category": "skills",
        "direction": "increase",
        "ordinal_order": [1, 2, 3, 4, 5],
    },
    "time_management_scale_1to5": {
        "display_name": "Time Management",
        "category": "skills",
        "direction": "increase",
        "ordinal_order": [1, 2, 3, 4, 5],
    },
    "stress_coping_yn": {
        "display_name": "Stress Coping",
        "category": "wellbeing",
        "direction": "activate",
        "ordinal_order": [0, 1],
    },
    "paid_classes_attending_yn": {
        "display_name": "Paid Classes Attendance",
        "category": "behavioral",
        "direction": "activate",
        "ordinal_order": [0, 1],
    },
    "academic_projects_completed_yn": {
        "display_name": "Academic Projects Completed",
        "category": "skills",
        "direction": "activate",
        "ordinal_order": [0, 1],
    },
    "internship_completion": {
        "display_name": "Internship Completion",
        "category": "skills",
        "direction": "activate",
        "ordinal_order": [0, 1],
    },
    "peer_group_quality_scale_1to5": {
        "display_name": "Peer Group Quality",
        "category": "skills",
        "direction": "increase",
        "ordinal_order": [1, 2, 3, 4, 5],
    },
    "socializing_frequency_scale_1to5": {
        "display_name": "Socializing Frequency",
        "category": "behavioral",
        "direction": "increase",
        "ordinal_order": [1, 2, 3, 4, 5],
    },
}

# === Contextual Features (partially modifiable, fallback for recommendations) ===
CONTEXTUAL_FEATURES = {
    "health_status_scale_1to5": {
        "display_name": "Health Status",
        "category": "wellbeing",
        "direction": "increase",
        "ordinal_order": [1, 2, 3, 4, 5],
    },
    "stress_frequency_scale_1to5": {
        "display_name": "Stress Frequency",
        "category": "wellbeing",
        "direction": "reduce",
        "ordinal_order": [1, 2, 3, 4, 5],
    },
    "internet_access": {
        "display_name": "Internet Access",
        "category": "context",
        "direction": "activate",
        "ordinal_order": [0, 1],
    },
    "alcohol_use": {
        "display_name": "Alcohol Use",
        "category": "behavioral",
        "direction": "reduce",
        "ordinal_order": [1, 0],
    },
    "tobacco_use": {
        "display_name": "Tobacco Use",
        "category": "behavioral",
        "direction": "reduce",
        "ordinal_order": [1, 0],
    },
    "other_addictions": {
        "display_name": "Other Addictions",
        "category": "behavioral",
        "direction": "reduce",
        "ordinal_order": [1, 0],
    },
}

# === Immutable Features ===
IMMUTABLE_FEATURES = [
    "gender", "age_group", "tenth_percentage", "twelth_percentage",
    "backlog_status", "residence_type", "institute_name", "course_branch",
    "mother_education_level", "mother_occupation",
    "father_education_level", "father_occupation",
    "family_size", "parental_marital_status",
    "family_income_bracket", "current_guardian",
    "nursery_education", "college_choice_reason", "course_choice_reason",
    "family_relationship_quality_scale_1to5",
    "commute_transport_mode", "commute_duration",
    "romantic_relationship_yn", "internship_payment_status",
    "family_extra_classes_motivation_status",
]

# === Ordinal Feature Encodings (for OrdinalEncoder) ===
ORDINAL_ENCODINGS = {
    "mother_education_level": ["No Education", "High School", "UG", "PG", "Any Other"],
    "father_education_level": ["No Education", "High School", "UG", "PG", "Any Other"],
    "family_income_bracket": [
        "Less than 2.5 Lakh", "2.5 Lakh to 8 Lakh",
        "8 Lakh to 15 Lakh", "More than 15 Lakh"
    ],
    "commute_duration": ["15 min", "15 min to 30 min", "30 min to 1 hour", "1 hour to 4 hour"],
    "daily_study_hours": ["15 to 30 min", "30 min to 1 hour", "1 hour - 4 hour", "More than 4 hours"],
}

# === Binary Columns (Yes/No mapping) ===
BINARY_COLUMNS = [
    "backlog_status", "nursery_education", "internet_access",
    "extracurricular_support", "workshop_seminar_participation_yn",
    "community_service_yn", "internship_completion",
    "academic_projects_completed_yn", "stress_coping_yn",
    "romantic_relationship_yn", "paid_classes_attending_yn",
    "alcohol_use", "tobacco_use", "other_addictions",
    "higher_education_interest_yn",
]

# === Ordinal Scale Columns (keep as-is, 1-5) ===
ORDINAL_SCALE_COLUMNS = [
    "family_relationship_quality_scale_1to5",
    "communication_skills_scale_1to5",
    "leadership_scale_1to5",
    "teamwork_skills_scale_1to5",
    "peer_group_quality_scale_1to5",
    "time_management_scale_1to5",
    "health_status_scale_1to5",
    "stress_frequency_scale_1to5",
    "socializing_frequency_scale_1to5",
]

# === Low-Variance Columns to Drop (>97% single class) ===
LOW_VARIANCE_DROP = ["age_group", "tobacco_use", "alcohol_use", "other_addictions"]

# === Feature Engineering: Study Hours Ordinal Map ===
STUDY_HOURS_ORDINAL = {
    "15 to 30 min": 1, "30 min to 1 hour": 2,
    "1 hour - 4 hour": 3, "More than 4 hours": 4,
}
STRESS_FREQ_ORDINAL = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

# === Optuna Configuration ===
OPTUNA_N_TRIALS = 50
OPTUNA_INNER_FOLDS = 3

# === FT-Transformer Default Hyperparameters ===
FT_D_TOKEN = 64
FT_N_BLOCKS = 2
FT_ATTENTION_DROPOUT = 0.2
FT_FFN_DROPOUT = 0.1
FT_RESIDUAL_DROPOUT = 0.0
FT_LR = 1e-3
FT_MAX_EPOCHS = 200
FT_PATIENCE = 20
FT_BATCH_SIZE = 128

# === SAINT Default Hyperparameters ===
SAINT_DIM = 32
SAINT_DEPTH = 3
SAINT_HEADS = 4
SAINT_ATTN_DROPOUT = 0.1
SAINT_FF_DROPOUT = 0.1
SAINT_LR = 1e-3
SAINT_MAX_EPOCHS = 200
SAINT_PATIENCE = 20
SAINT_BATCH_SIZE = 128

# === Optuna Search Spaces ===
OPTUNA_SEARCH_SPACES = {
    "LogisticRegression": {
        "C": (0.01, 100.0, "log"),
        "solver": ["lbfgs", "saga"],
    },
    "RandomForest": {
        "n_estimators": (50, 500),
        "max_depth": (3, 20),
        "min_samples_leaf": (1, 20),
    },
    "SVM": {
        "C": (0.01, 100.0, "log"),
        "kernel": ["rbf", "poly"],
        "gamma": (1e-4, 1.0, "log"),
    },
    "XGBoost": {
        "n_estimators": (50, 500),
        "max_depth": (3, 10),
        "learning_rate": (0.01, 0.3, "log"),
        "subsample": (0.5, 1.0),
    },
    "TabNet": {
        "n_d": (8, 64),
        "n_steps": (3, 7),
        "lambda_sparse": (1e-4, 1e-2, "log"),
        "lr": (1e-3, 5e-2, "log"),
    },
    "FTTransformer": {
        "n_blocks": (1, 4),
        "d_token": (32, 128),
        "attention_dropout": (0.0, 0.3),
        "ffn_dropout": (0.0, 0.3),
        "lr": (1e-4, 1e-2, "log"),
    },
    "SAINT": {
        "depth": (1, 6),
        "heads": (2, 8),
        "dim": (16, 128),
        "attn_dropout": (0.0, 0.3),
        "lr": (1e-4, 1e-2, "log"),
    },
}

# === Baseline Model Names ===
BASELINE_MODELS = ["LogisticRegression", "RandomForest", "SVM", "XGBoost"]
DL_MODELS = ["TabNet", "FTTransformer", "SAINT"]
ALL_MODELS = BASELINE_MODELS + DL_MODELS

# === Objectives (data-driven: placement dropped due to 7.1% imbalance and no signal) ===
OBJECTIVES = ["cgpa_category", "at_risk", "career_ready"]

# === At-Risk Definition ===
AT_RISK_CGPA_THRESHOLD = 7.0  # CGPA < 7.0 OR has backlogs = at-risk
