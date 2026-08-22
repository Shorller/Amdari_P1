# 🩺 NutriGlyc AI — Glucose Spike Prediction & Model Explorer

An end-to-end machine learning project for predicting post-meal glucose spike events from dietary, clinical and lifestyle characteristics.

The project covers data quality assessment, exploratory data analysis, feature engineering, model comparison, hyperparameter tuning, model explainability using SHAP, and deployment through an interactive Streamlit application.

> **Important:** This is a data science portfolio project and model exploration tool. It is not a medical device and must not be used for diagnosis, treatment, insulin dosing, dietary decisions or other healthcare decisions.

---

## 📌 Project Overview

NutriGlyc AI was developed as part of a data science internship project to explore whether information available around a person's meal, clinical characteristics and lifestyle could be used to distinguish between glucose spike and non-spike events.

The project had three main objectives:

1. Build a machine learning model capable of classifying glucose spike events.
2. Identify the variables contributing most strongly to model predictions.
3. Translate the trained model into an interactive application for demonstrating how machine learning predictions respond to hypothetical inputs.

A major consideration throughout the project was preventing **data leakage** so that the model relied only on information that could reasonably be available before the outcome.

---

## 📊 Dataset

The original dataset contained **5,150 records and 28 variables** covering:

- Demographic characteristics
- Clinical measurements
- Meal composition
- Nutritional indicators
- Lifestyle characteristics
- Glucose measurements

The target variable, `glucose_spike`, indicates whether a post-meal glucose spike occurred.

### Data quality assessment

Initial assessment identified:

| Quality Check | Finding |
| --- | ---: |
| Original records | 5,150 |
| Original variables | 28 |
| Missing values | 1,243 |
| Duplicate records | 150 |
| Variables containing outliers | 16 |
| Invalid categorical values | 0 |
| Numeric range violations | 0 |

After cleaning, the final dataset contained **5,000 records with no missing values or duplicate records**.

The target was reasonably balanced:

- **No Spike:** 2,683 observations (53.7%)
- **Spike:** 2,317 observations (46.3%)

---

## 🧹 Data Preparation

The preparation pipeline included:

- Duplicate removal
- Median imputation for missing numerical values
- IQR-based winsorisation of extreme values
- Categorical encoding
- Feature engineering
- Stratified train-test splitting
- Feature scaling

The data was split into:

- **Training set:** 4,000 observations
- **Test set:** 1,000 observations

Scaling was fitted exclusively on the training data before being applied to the test set to prevent information leakage.

---

## 🔍 Exploratory Data Analysis

Exploratory analysis identified several strong relationships between the available variables and glucose spike occurrence.

Some of the clearest associations were found for:

- Carbohydrate intake
- Glycaemic load
- Carbohydrate-to-fibre ratio
- Meal risk score
- Physical activity
- Insulin-related variables

For example, median carbohydrate intake was approximately **156.2g among spike events**, compared with **90.1g among non-spike events**.

The analysis also identified two variables with particularly strong correlations with the target:

- `glucose_change`
- `post_meal_glucose`

However, these variables were not retained for modelling.

---

## 🚫 Preventing Data Leakage

`post_meal_glucose` and `glucose_change` are derived from information available after the meal outcome.

Including them would allow the model to indirectly observe information about the event it was supposed to predict.

Both variables were therefore removed before model training.

This ensured that model performance was evaluated using information that could plausibly exist before the outcome rather than information derived from it.

---

## 🛠️ Feature Engineering

Additional features were created to capture relationships not fully represented by the original variables.

These included:

- `high_carb_flag`
- `high_gl_flag`
- `low_activity_flag`
- `sugar_carb_ratio`
- `bmi_category`
- `insulin_carb_ratio`
- `poor_sleep_flag`
- `high_stress_flag`

The engineered **insulin-to-carbohydrate ratio** subsequently became one of the most influential features in the final model.

The original modelling pipeline contained **32 predictor variables**.

---

## 🤖 Model Development

Three main classification approaches were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

XGBoost was subsequently tuned using `RandomizedSearchCV` with stratified 5-fold cross-validation.

### Model comparison

| Model | Accuracy | Recall | F1 Score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.775 | 0.7775 | 0.7619 | **0.8540** |
| Random Forest | 0.768 | 0.7646 | 0.7532 | 0.8419 |
| **XGBoost (Tuned)** | **0.776** | **0.8315** | **0.7746** | 0.8531 |

Logistic Regression produced a marginally higher ROC-AUC, while tuned XGBoost achieved the strongest **recall and F1 score**.

XGBoost correctly identified **385 of 463 spike events**, reducing false negatives to **78**, compared with 103 for Logistic Regression and 109 for Random Forest.

For this project, the tuned XGBoost model was therefore selected as the final modelling approach.

---

## ⚙️ XGBoost Tuning

The tuning process explored parameters including:

- Number of estimators
- Maximum tree depth
- Learning rate
- Subsampling
- Column sampling
- Minimum child weight
- Gamma
- Class weighting
- L1 regularisation
- L2 regularisation

The model was evaluated using stratified 5-fold cross-validation.

The selected model showed stable performance across folds:

- **Mean CV ROC-AUC:** 0.8445
- **Standard deviation:** 0.0113

---

## 🧠 Model Explainability with SHAP

SHAP was used to investigate how strongly individual features contributed to XGBoost predictions.

The five most influential variables by mean absolute SHAP value were:

| Rank | Feature | SHAP Importance |
| --- | --- | ---: |
| 1 | Carbohydrate intake | 32.4% |
| 2 | Insulin-to-carbohydrate ratio | 16.9% |
| 3 | Glycaemic load | 11.7% |
| 4 | Physical activity | 9.4% |
| 5 | Stress level | 5.5% |

The **top 10 features accounted for approximately 87.9% of total SHAP importance**.

One particularly useful result from the feature-engineering process was the `insulin_carb_ratio`, which became the second most influential variable according to SHAP.

SHAP importance measures contribution to model predictions; it should not be interpreted as evidence that a feature causes glucose spikes or provides a clinical treatment target.

---

## 🚀 Deployment Model

The interactive application uses a slightly modified version of the analytical model.

The original model contained **32 predictors**, including `meal_risk_score`. Because the dataset did not provide a reproducible formula for calculating `meal_risk_score` from raw user inputs, this variable was removed before deployment.

The deployment model was therefore retrained and tuned using **31 reproducible features**.

### Original vs deployment model

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original XGBoost — 32 features | 0.776 | 0.7250 | 0.8315 | 0.7746 | 0.8531 |
| Deployment XGBoost — 31 features | 0.766 | 0.7181 | 0.8143 | 0.7632 | 0.8528 |

Removing `meal_risk_score` resulted in only a very small reduction in ROC-AUC while making every feature required by the deployed model reproducible from the application's inputs.

This separation also avoids inventing or approximating an undocumented feature calculation during deployment.

---

## 🖥️ Interactive Model Explorer

A Streamlit application was developed to allow users to explore how the deployment model responds to hypothetical combinations of:

- Demographic characteristics
- Meal composition
- Glycaemic index
- Clinical characteristics
- Physical activity
- Sleep
- Stress
- Smoking and alcohol indicators

The user provides understandable raw values while the application automatically generates several engineered variables required by the model.

For example:

```python
glycemic_load = (glycemic_index * carb_intake) / 100
carb_fiber_ratio = carb_intake / max(fiber_intake, 1e-5)
sugar_carb_ratio = sugar_intake / (carb_intake + 1e-5)
insulin_carb_ratio = insulin_dose / (carb_intake + 1e-5)
```

The application then reproduces the preprocessing pipeline used during training before passing the observation to the trained XGBoost model.

The interface reports:

- Predicted spike probability
- Predicted model class
- Selected derived model features

The application is intentionally presented as a **model exploration prototype rather than a clinical decision-support system**.

---

## ⚠️ Limitations

Several limitations should be considered when interpreting this project:

- The model was developed from the project's supplied dataset and has not been clinically validated.
- Predictive relationships do not establish causality.
- Model performance may not generalise to different populations or real-world clinical environments.
- Median imputation created visible concentrations around the median for some variables.
- Some engineered variables in the original dataset did not have fully documented derivation methods.
- The deployment model therefore excludes `meal_risk_score`.
- Model probabilities should not be interpreted as individual medical risk estimates.

This project is intended to demonstrate **data preparation, machine learning, explainability and deployment techniques**.

---

## 🧰 Tools & Technologies

**Data analysis**

`Python` `pandas` `NumPy`

**Machine learning**

`scikit-learn` `XGBoost` `RandomizedSearchCV`

**Model explainability**

`SHAP`

**Visualisation**

`Matplotlib` `Seaborn`

**Deployment**

`Streamlit` `joblib`

**Development**

`Jupyter Notebook` `VS Code` `Git` `GitHub`

---

## 📁 Repository Structure

```text
Amdari_P1/
│
├── app.py
├── README.md
├── requirements.txt
│
├── *.ipynb
│   └── Project analysis and modelling notebooks
│
├── *.pkl
│   └── Deployment model and preprocessing assets
│
└── data/
    └── Project dataset
```

The exact repository structure may evolve as the project is organised for deployment.

---

## ▶️ Running the Application Locally

Clone the repository:

```bash
git clone https://github.com/Shorller/Amdari_P1.git
cd Amdari_P1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 🔗 Links

**GitHub repository:**  
https://github.com/Shorller/Amdari_P1

**Live Streamlit application:**  
*Link will be added after deployment.*

---

## 👤 Author

**Oluwashola Rufai**

Data science portfolio project developed as part of a data science internship.

---

## Disclaimer

This repository and its Streamlit application are provided for educational, portfolio and machine-learning demonstration purposes only.

The model has **not been clinically validated** and is **not a medical device**. Its outputs must not be used for diagnosis, treatment, medication management, insulin dosing, dietary decisions or other healthcare decisions.