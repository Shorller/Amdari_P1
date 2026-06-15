# ============================================================
# setup_repo.py
# NutriGlyc AI Solutions — Repository Setup Utility
# Run once from terminal: python setup_repo.py
# ============================================================

import os
import shutil
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score
import xgboost as xgb

# ============================================================
# CONFIGURATION
# ============================================================

BASE_PATH = (
    r'C:\Users\Lovel\OneDrive - University of Glasgow'
    r'\Documents\Amdari_P1'
)

DATASET_PATH = os.path.join(
    BASE_PATH, 'Glucose_Spike_Dataset.xlsx'
)

# ============================================================
# STEP 1 — CREATE FOLDER STRUCTURE
# ============================================================

def create_folders():
    print("=" * 60)
    print("STEP 1 — CREATING FOLDER STRUCTURE")
    print("=" * 60)

    folders = [
        'notebooks',
        'data/raw',
        'data/processed',
        'models',
        'reports/figures',
        'reports/summaries',
        'src'
    ]

    for folder in folders:
        full_path = os.path.join(BASE_PATH, folder)
        os.makedirs(full_path, exist_ok=True)
        print(f"  ✅ {folder}")

    print(f"\n✅ Folder structure created.")

# ============================================================
# STEP 2 — COPY FILES TO CORRECT LOCATIONS
# ============================================================

def organise_files():
    print("\n" + "=" * 60)
    print("STEP 2 — ORGANISING FILES")
    print("=" * 60)

    file_moves = [
        (
            os.path.join(BASE_PATH, 'Glucose_Spike_Dataset.xlsx'),
            os.path.join(BASE_PATH, 'data', 'raw',
                         'Glucose_Spike_Dataset.xlsx')
        ),
        (
            os.path.join(BASE_PATH, 'glucose_spike.ipynb'),
            os.path.join(BASE_PATH, 'notebooks',
                         'glucose_spike.ipynb')
        )
    ]

    for src, dst in file_moves:
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  ✅ Copied: {os.path.basename(src)}"
                  f" → {os.path.relpath(dst, BASE_PATH)}")
        elif os.path.exists(dst):
            print(f"  ℹ️  Already exists: "
                  f"{os.path.relpath(dst, BASE_PATH)}")
        else:
            print(f"  ⚠️  Source not found: "
                  f"{os.path.basename(src)}")

    print(f"\n✅ Files organised.")

# ============================================================
# STEP 3 — REBUILD PIPELINE & TRAIN MODELS
# ============================================================

def build_pipeline():
    print("\n" + "=" * 60)
    print("STEP 3 — REBUILDING PIPELINE")
    print("=" * 60)

    # Load
    print("\n  Loading dataset...")
    data     = pd.read_excel(DATASET_PATH)
    data_raw = data.copy()
    print(f"  ✅ Loaded: {data_raw.shape}")

    # Clean
    print("  Cleaning data...")
    df = data_raw.copy()
    df = df.drop_duplicates().reset_index(drop=True)

    numeric_cols     = df.select_dtypes(
        include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=['object']).columns.tolist()
    exclude          = ['patient_id', 'glucose_spike',
                        'medication_adherence']

    for col in [c for c in numeric_cols if c not in exclude]:
        df[col] = df[col].fillna(df[col].median())
    for col in [c for c in categorical_cols
                if c not in exclude]:
        df[col] = df[col].fillna(df[col].mode()[0])

    binary_cols = ['medication_adherence', 'glucose_spike']
    for col in [c for c in numeric_cols
                if c not in ['patient_id'] + binary_cols]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(
            lower=Q1 - 1.5 * IQR,
            upper=Q3 + 1.5 * IQR
        )
    print(f"  ✅ Cleaned: {df.shape}")

    # Feature engineering
    print("  Engineering features...")
    df = df.drop(columns=['post_meal_glucose',
                           'glucose_change',
                           'patient_id'])

    binary_map = {
        'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0
    }
    for col in ['gender', 'smoking_status',
                'alcohol_consumption']:
        df[col] = df[col].map(binary_map)

    label_encoders = {}
    for col in ['diabetes_type', 'meal_time']:
        le           = LabelEncoder()
        df[col]      = le.fit_transform(df[col])
        label_encoders[col] = le

    carb_75 = df['carb_intake'].quantile(0.75)
    gl_75   = df['glycemic_load'].quantile(0.75)

    df['high_carb_flag']     = (
        df['carb_intake'] > carb_75).astype(int)
    df['high_gl_flag']       = (
        df['glycemic_load'] > gl_75).astype(int)
    df['low_activity_flag']  = (
        df['physical_activity'] < 30).astype(int)
    df['sugar_carb_ratio']   = (
        df['sugar_intake'] / (df['carb_intake'] + 1e-5))
    df['bmi_category']       = df['bmi'].apply(
        lambda x: 0 if x < 18.5 else 1 if x < 25
        else 2 if x < 30 else 3)
    df['insulin_carb_ratio'] = (
        df['insulin_dose'] / (df['carb_intake'] + 1e-5))
    df['poor_sleep_flag']    = (
        df['sleep_hours'] < 6).astype(int)
    df['high_stress_flag']   = (
        df['stress_level'] > 7).astype(int)

    print(f"  ✅ Features engineered: {df.shape}")

    # Split
    X = df.drop(columns=['glucose_spike'])
    y = df['glucose_spike']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y
    )

    # Scale
    do_not_scale = [
        'gender', 'smoking_status', 'alcohol_consumption',
        'diabetes_type', 'meal_time', 'medication_adherence',
        'high_carb_flag', 'high_gl_flag', 'low_activity_flag',
        'poor_sleep_flag', 'high_stress_flag', 'bmi_category'
    ]
    scale_cols = [c for c in X_train.columns
                  if c not in do_not_scale]

    scaler         = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()
    X_train_scaled[scale_cols] = scaler.fit_transform(
        X_train[scale_cols])
    X_test_scaled[scale_cols]  = scaler.transform(
        X_test[scale_cols])

    print("  ✅ Scaling applied")

    # Train models
    print("\n  Training models...")

    lr_model = LogisticRegression(
        max_iter=1000, random_state=42, solver='lbfgs')
    lr_model.fit(X_train_scaled, y_train)
    lr_acc = accuracy_score(
        y_test, lr_model.predict(X_test_scaled))
    print(f"  ✅ Logistic Regression — Accuracy: {lr_acc:.4f}")

    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    rf_acc = accuracy_score(
        y_test, rf_model.predict(X_test_scaled))
    print(f"  ✅ Random Forest       — Accuracy: {rf_acc:.4f}")

    xgb_tuned = xgb.XGBClassifier(
        n_estimators      = 100,
        max_depth         = 3,
        learning_rate     = 0.05,
        subsample         = 0.7,
        colsample_bytree  = 0.7,
        min_child_weight  = 1,
        gamma             = 0.1,
        scale_pos_weight  = 1.2,
        reg_lambda        = 1.5,
        reg_alpha         = 0,
        random_state      = 42,
        eval_metric       = 'logloss',
        verbosity         = 0,
        use_label_encoder = False
    )
    xgb_tuned.fit(X_train_scaled, y_train)
    xgb_acc = accuracy_score(
        y_test, xgb_tuned.predict(X_test_scaled))
    print(f"  ✅ XGBoost Tuned       — Accuracy: {xgb_acc:.4f}")

    return (lr_model, rf_model, xgb_tuned,
            scaler, label_encoders,
            X_test_scaled, y_test)

# ============================================================
# STEP 4 — SAVE MODELS
# ============================================================

def save_models(lr_model, rf_model, xgb_tuned,
                scaler, label_encoders):
    print("\n" + "=" * 60)
    print("STEP 4 — SAVING MODELS")
    print("=" * 60)

    models_path = os.path.join(BASE_PATH, 'models')

    joblib.dump(lr_model,
                os.path.join(models_path,
                             'logistic_regression.pkl'))
    print("  ✅ logistic_regression.pkl")

    joblib.dump(rf_model,
                os.path.join(models_path,
                             'random_forest.pkl'))
    print("  ✅ random_forest.pkl")

    joblib.dump(xgb_tuned,
                os.path.join(models_path,
                             'xgboost_tuned.pkl'))
    print("  ✅ xgboost_tuned.pkl")

    joblib.dump(scaler,
                os.path.join(models_path,
                             'standard_scaler.pkl'))
    print("  ✅ standard_scaler.pkl")

    joblib.dump(label_encoders,
                os.path.join(models_path,
                             'label_encoders.pkl'))
    print("  ✅ label_encoders.pkl")

    print(f"\n✅ All models saved → models/")

# ============================================================
# STEP 5 — SAVE REPORT SUMMARIES
# ============================================================

def save_reports(lr_model, rf_model, xgb_tuned,
                 X_test_scaled, y_test):
    print("\n" + "=" * 60)
    print("STEP 5 — SAVING REPORT SUMMARIES")
    print("=" * 60)

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, roc_curve,
        precision_recall_curve
    )
    import shap

    reports_path = os.path.join(
        BASE_PATH, 'reports', 'summaries'
    )

    # Model comparison
    models_dict = {
        'Logistic Regression' : lr_model,
        'Random Forest'       : rf_model,
        'XGBoost (Tuned)'     : xgb_tuned
    }

    rows = []
    for name, model in models_dict.items():
        pred  = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]
        rows.append({
            'Model'    : name,
            'Accuracy' : round(accuracy_score(y_test, pred), 4),
            'Precision': round(precision_score(y_test, pred), 4),
            'Recall'   : round(recall_score(y_test, pred), 4),
            'F1-Score' : round(f1_score(y_test, pred), 4),
            'ROC-AUC'  : round(roc_auc_score(y_test, proba), 4)
        })

    full_results = pd.DataFrame(rows)
    full_results.to_csv(
        os.path.join(reports_path, 'model_comparison.csv'),
        index=False
    )
    print("  ✅ model_comparison.csv")

    # SHAP importance
    print("  Computing SHAP values...")
    explainer   = shap.TreeExplainer(xgb_tuned)
    shap_values = explainer.shap_values(X_test_scaled)
    shap_df     = pd.DataFrame(
        shap_values,
        columns=X_test_scaled.columns.tolist()
    )

    mean_abs_shap = np.abs(shap_df).mean().sort_values(
        ascending=False).reset_index()
    mean_abs_shap.columns = ['Feature', 'Mean |SHAP|']
    mean_abs_shap['Rank'] = range(1, len(mean_abs_shap) + 1)
    mean_abs_shap['Importance (%)'] = (
        mean_abs_shap['Mean |SHAP|'] /
        mean_abs_shap['Mean |SHAP|'].sum() * 100
    ).round(2)
    mean_abs_shap['Direction'] = mean_abs_shap[
        'Feature'].apply(
        lambda f: 'Increases Risk'
        if shap_df[f].mean() >= 0 else 'Reduces Risk'
    )
    mean_abs_shap.to_csv(
        os.path.join(reports_path,
                     'shap_feature_importance.csv'),
        index=False
    )
    print("  ✅ shap_feature_importance.csv")

    # Threshold analysis
    xgb_proba       = xgb_tuned.predict_proba(
        X_test_scaled)[:, 1]
    fpr, tpr, thrs  = roc_curve(y_test, xgb_proba)
    youden_thresh   = float(thrs[np.argmax(tpr - fpr)])

    prec_v, rec_v, pr_thrs = precision_recall_curve(
        y_test, xgb_proba)
    f1_v         = (2 * prec_v[:-1] * rec_v[:-1]) / \
                   (prec_v[:-1] + rec_v[:-1] + 1e-8)
    f1_thresh    = float(pr_thrs[np.argmax(f1_v)])

    thresh_rows = []
    for label, thresh in [
        ('Default (0.50)', 0.50),
        ("Youden's J",     round(youden_thresh, 4)),
        ('F1 Optimised',   round(f1_thresh, 4))
    ]:
        p_t = (xgb_proba >= thresh).astype(int)
        thresh_rows.append({
            'Threshold Label': label,
            'Threshold'      : thresh,
            'Accuracy'  : round(accuracy_score(y_test, p_t), 4),
            'Precision' : round(precision_score(y_test, p_t), 4),
            'Recall'    : round(recall_score(y_test, p_t), 4),
            'F1-Score'  : round(f1_score(y_test, p_t), 4),
            'ROC-AUC'   : round(
                roc_auc_score(y_test, xgb_proba), 4)
        })

    pd.DataFrame(thresh_rows).to_csv(
        os.path.join(reports_path, 'threshold_analysis.csv'),
        index=False
    )
    print("  ✅ threshold_analysis.csv")
    print(f"\n✅ All reports saved → reports/summaries/")

# ============================================================
# STEP 6 — GENERATE README.md
# ============================================================

def generate_readme():
    print("\n" + "=" * 60)
    print("STEP 6 — GENERATING README.md")
    print("=" * 60)

    lines = [
        "# Glucose Spike Prediction & Nutrition Risk Analytics System",
        "",
        "**Company:** NutriGlyc AI Solutions",
        "**Location:** California, USA",
        "**Environment:** Python 3.13 | Jupyter Notebook | VS Code",
        "",
        "---",
        "",
        "## Project Overview",
        "",
        "This project develops an end-to-end machine learning pipeline",
        "to predict postprandial (after-meal) blood glucose spikes in",
        "diabetic patients using clinical, dietary, and lifestyle data.",
        "The system combines exploratory data analysis, feature",
        "engineering, and predictive modelling to identify high-risk",
        "patients and support early preventive intervention.",
        "",
        "---",
        "",
        "## Objectives",
        "",
        "1. Predict the likelihood of postprandial glucose spikes using",
        "   patient health and dietary indicators",
        "2. Identify the primary nutritional and clinical risk factors",
        "   driving glucose spike occurrence",
        "3. Support healthcare professionals and individuals with",
        "   data-driven preventive nutrition recommendations",
        "",
        "---",
        "",
        "## Dataset",
        "",
        "| Detail | Value |",
        "|---|---|",
        "| File | Glucose_Spike_Dataset.xlsx |",
        "| Raw Rows | 5,150 |",
        "| Cleaned Rows | 5,000 |",
        "| Columns | 28 |",
        "| Target | glucose_spike (1 = Spike, 0 = No Spike) |",
        "| Class Balance | 53.7% No Spike / 46.3% Spike |",
        "",
        "---",
        "",
        "## Repository Structure",
        "",
        "    Amdari_P1/",
        "    |",
        "    |-- notebooks/",
        "    |   |-- glucose_spike.ipynb",
        "    |",
        "    |-- data/",
        "    |   |-- raw/",
        "    |   |   |-- Glucose_Spike_Dataset.xlsx",
        "    |   |-- processed/",
        "    |",
        "    |-- models/",
        "    |   |-- logistic_regression.pkl",
        "    |   |-- random_forest.pkl",
        "    |   |-- xgboost_tuned.pkl",
        "    |   |-- standard_scaler.pkl",
        "    |   |-- label_encoders.pkl",
        "    |",
        "    |-- reports/",
        "    |   |-- figures/",
        "    |   |-- summaries/",
        "    |       |-- model_comparison.csv",
        "    |       |-- shap_feature_importance.csv",
        "    |       |-- threshold_analysis.csv",
        "    |",
        "    |-- src/",
        "    |-- setup_repo.py",
        "    |-- README.md",
        "    |-- requirements.txt",
        "",
        "---",
        "",
        "## Pipeline Stages",
        "",
        "| Stage | Description |",
        "|---|---|",
        "| 1 | Library Imports & Data Loading |",
        "| 2 | Data Quality Assessment |",
        "| 3 | Data Cleaning & Preprocessing |",
        "| 4 | Exploratory Data Analysis |",
        "| 5 | Transform & Integrate Data |",
        "| 6 | Feature Engineering |",
        "| 7 | Baseline Model Training |",
        "| 8 | Validate Results & Extract Insights |",
        "| 9 | XGBoost Hyperparameter Tuning |",
        "| 10 | Model Evaluation & Threshold Selection |",
        "| 11 | SHAP Insights & Recommendations |",
        "",
        "---",
        "",
        "## Model Results",
        "",
        "| Model | Accuracy | Recall | F1-Score | ROC-AUC |",
        "|---|---|---|---|---|",
        "| Logistic Regression | 77.50% | 0.7775 | 0.7619 | 0.8540 |",
        "| Random Forest | 76.80% | 0.7646 | 0.7532 | 0.8419 |",
        "| XGBoost Default | 73.00% | 0.7149 | 0.7103 | 0.8176 |",
        "| XGBoost Tuned | 77.60% | 0.8315 | 0.7746 | 0.8531 |",
        "",
        "### Selected Model: XGBoost Tuned",
        "- Optimal threshold: 0.50 (confirmed via Youden's J)",
        "- Reason: Highest recall (0.8315) — fewest missed spikes",
        "",
        "### Best Hyperparameters",
        "",
        "| Parameter | Value |",
        "|---|---|",
        "| n_estimators | 100 |",
        "| max_depth | 3 |",
        "| learning_rate | 0.05 |",
        "| subsample | 0.7 |",
        "| colsample_bytree | 0.7 |",
        "| scale_pos_weight | 1.2 |",
        "| gamma | 0.1 |",
        "| reg_lambda | 1.5 |",
        "",
        "---",
        "",
        "## Top 10 Risk Factors (SHAP Analysis)",
        "",
        "| Rank | Feature | Direction |",
        "|---|---|---|",
        "| 1 | carb_intake | Increases spike risk |",
        "| 2 | glycemic_load | Increases spike risk |",
        "| 3 | insulin_carb_ratio | Reduces spike risk |",
        "| 4 | meal_risk_score | Increases spike risk |",
        "| 5 | carb_fiber_ratio | Increases spike risk |",
        "| 6 | sugar_carb_ratio | Reduces spike risk |",
        "| 7 | physical_activity | Reduces spike risk |",
        "| 8 | insulin_dose | Reduces spike risk |",
        "| 9 | high_carb_flag | Increases spike risk |",
        "| 10 | glycemic_index | Increases spike risk |",
        "",
        "---",
        "",
        "## Key Clinical Recommendations",
        "",
        "1. Limit carbohydrate intake to below 90g per meal",
        "2. Choose low glycemic load foods (GL < 10 per serving)",
        "3. Calibrate insulin-to-carb ratios with a healthcare provider",
        "4. Increase dietary fibre to at least 25-30g per day",
        "5. Aim for 30+ minutes of physical activity daily",
        "",
        "---",
        "",
        "## Reproduction Steps",
        "",
        "1. Clone the repository",
        "   git clone https://github.com/Shorller/Amdari_P1.git",
        "",
        "2. Install dependencies",
        "   pip install -r requirements.txt",
        "",
        "3. Run setup script",
        "   python setup_repo.py",
        "",
        "4. Open the notebook",
        "   Open notebooks/glucose_spike.ipynb and run all cells",
        "",
        "5. Load saved model",
        "   import joblib",
        "   model  = joblib.load('models/xgboost_tuned.pkl')",
        "   scaler = joblib.load('models/standard_scaler.pkl')",
        "",
        "---",
        "",
        "## Requirements",
        "",
        "pandas, numpy, matplotlib, seaborn,",
        "scikit-learn, xgboost, shap, joblib, openpyxl",
        "",
        "---",
        "",
        "## Author",
        "",
        "Oluwashola",
        "NutriGlyc AI Solutions — Data Science Project",
        "University of Glasgow",
    ]

    readme_content = "\n".join(lines)

    with open(os.path.join(BASE_PATH, 'README.md'),
              'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(" ✅ README.md generated")   

































    