import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NutriGlyc AI | Model Explorer",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# LOAD DEPLOYMENT ASSETS
# ============================================================

@st.cache_resource
def load_assets():

    model = joblib.load(
        "nutriglyc_deployment_xgboost.pkl"
    )

    scaler = joblib.load(
        "nutriglyc_deployment_scaler.pkl"
    )

    features = joblib.load(
        "nutriglyc_deployment_features.pkl"
    )

    scale_cols = joblib.load(
        "nutriglyc_deployment_scale_cols.pkl"
    )

    label_encoders = joblib.load(
        "nutriglyc_label_encoders.pkl"
    )

    binary_map = joblib.load(
        "nutriglyc_binary_map.pkl"
    )

    thresholds = joblib.load(
        "nutriglyc_thresholds.pkl"
    )

    return (
        model,
        scaler,
        features,
        scale_cols,
        label_encoders,
        binary_map,
        thresholds
    )


(
    model,
    scaler,
    feature_order,
    scale_cols,
    label_encoders,
    binary_map,
    thresholds
) = load_assets()


# ============================================================
# HEADER
# ============================================================

st.title("🩺 NutriGlyc AI")

st.subheader(
    "Glucose Spike Prediction Model Explorer"
)

st.write(
    "Explore how a trained machine-learning model responds to "
    "different example dietary, clinical and lifestyle inputs."
)

st.info(
    "This application is a data-science demonstration. "
    "It is not a medical device and its predictions should not "
    "be used for diagnosis, treatment, insulin dosing, dietary "
    "decisions or other healthcare decisions."
)


# ============================================================
# TABS
# ============================================================

prediction_tab, about_tab = st.tabs(
    ["Prediction Explorer", "About the Model"]
)


# ============================================================
# PREDICTION EXPLORER
# ============================================================

with prediction_tab:

    st.header("Example inputs")

    st.caption(
        "Enter hypothetical values to explore how the model responds."
    )

    # --------------------------------------------------------
    # DEMOGRAPHIC / BACKGROUND
    # --------------------------------------------------------

    st.subheader("Background characteristics")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=45
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

    with col2:

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=27.0,
            step=0.1
        )

        diabetes_type = st.selectbox(
            "Diabetes type",
            ["Type 1", "Type 2"]
        )

    with col3:

        medication_adherence = st.selectbox(
            "Medication adherence",
            ["No", "Yes"]
        )

        pre_meal_glucose = st.number_input(
            "Pre-meal glucose",
            min_value=40.0,
            max_value=300.0,
            value=110.0,
            step=1.0,
            help="Example model input from the project dataset."
        )


    # --------------------------------------------------------
    # MEAL CHARACTERISTICS
    # --------------------------------------------------------

    st.divider()

    st.subheader("Meal characteristics")

    col1, col2, col3 = st.columns(3)

    with col1:

        meal_time = st.selectbox(
            "Meal time",
            ["Breakfast", "Lunch", "Dinner", "Snack"]
        )

        carb_intake = st.number_input(
            "Carbohydrate intake (g)",
            min_value=0.0,
            max_value=400.0,
            value=80.0,
            step=1.0
        )

        protein_intake = st.number_input(
            "Protein intake (g)",
            min_value=0.0,
            max_value=200.0,
            value=30.0,
            step=1.0
        )

    with col2:

        fat_intake = st.number_input(
            "Fat intake (g)",
            min_value=0.0,
            max_value=200.0,
            value=25.0,
            step=1.0
        )

        fiber_intake = st.number_input(
            "Fiber intake (g)",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=1.0
        )

        sugar_intake = st.number_input(
            "Sugar intake (g)",
            min_value=0.0,
            max_value=200.0,
            value=20.0,
            step=1.0
        )

    with col3:

        glycemic_index = st.number_input(
            "Glycemic Index",
            min_value=0.0,
            max_value=100.0,
            value=55.0,
            step=1.0
        )

        portion_size = st.number_input(
            "Portion size",
            min_value=0.0,
            max_value=1000.0,
            value=300.0,
            step=10.0,
            help="Enter a hypothetical portion-size value consistent with the project dataset."
        )

        water_intake = st.number_input(
            "Water intake",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Example model input from the project dataset."
        )


    # --------------------------------------------------------
    # LIFESTYLE / CLINICAL
    # --------------------------------------------------------

    st.divider()

    st.subheader("Lifestyle and clinical characteristics")

    col1, col2, col3 = st.columns(3)

    with col1:

        insulin_dose = st.number_input(
            "Insulin dose",
            min_value=0.0,
            max_value=100.0,
            value=3.0,
            step=0.5,
            help="Hypothetical model input only. This application must not be used to determine insulin dosage."
        )

        physical_activity = st.number_input(
            "Physical activity (minutes)",
            min_value=0.0,
            max_value=300.0,
            value=45.0,
            step=5.0
        )

    with col2:

        stress_level = st.number_input(
            "Stress level",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=1.0
        )

        sleep_hours = st.number_input(
            "Sleep hours",
            min_value=0.0,
            max_value=15.0,
            value=7.0,
            step=0.5
        )

    with col3:

        smoking_status = st.selectbox(
            "Smoking status",
            ["No", "Yes"]
        )

        alcohol_consumption = st.selectbox(
            "Alcohol consumption",
            ["No", "Yes"]
        )


    # ========================================================
    # PREPARE INPUT
    # ========================================================

    def prepare_input():

        # ----------------------------------------------------
        # Original categorical encodings
        # ----------------------------------------------------

        gender_encoded = binary_map[gender]

        smoking_encoded = binary_map[smoking_status]

        alcohol_encoded = binary_map[alcohol_consumption]

        medication_encoded = (
            1 if medication_adherence == "Yes" else 0
        )

        diabetes_encoded = int(
            label_encoders["diabetes_type"].transform(
                [diabetes_type]
            )[0]
        )

        meal_encoded = int(
            label_encoders["meal_time"].transform(
                [meal_time]
            )[0]
        )


        # ----------------------------------------------------
        # AUTOMATICALLY DERIVED MODEL FEATURES
        # ----------------------------------------------------

        glycemic_load = (
            glycemic_index * carb_intake
        ) / 100

        carb_fiber_ratio = (
            carb_intake / max(fiber_intake, 1e-5)
        )

        sugar_carb_ratio = (
            sugar_intake / (carb_intake + 1e-5)
        )

        insulin_carb_ratio = (
            insulin_dose / (carb_intake + 1e-5)
        )


        # ----------------------------------------------------
        # ENGINEERED FLAGS
        # ----------------------------------------------------

        high_carb_flag = int(
            carb_intake > thresholds["carb_75"]
        )

        high_gl_flag = int(
            glycemic_load > thresholds["gl_75"]
        )

        low_activity_flag = int(
            physical_activity < 30
        )

        poor_sleep_flag = int(
            sleep_hours < 6
        )

        high_stress_flag = int(
            stress_level > 7
        )


        # ----------------------------------------------------
        # BMI CATEGORY
        # ----------------------------------------------------

        if bmi < 18.5:
            bmi_category = 0

        elif bmi < 25:
            bmi_category = 1

        elif bmi < 30:
            bmi_category = 2

        else:
            bmi_category = 3


        # ----------------------------------------------------
        # COMPLETE MODEL INPUT
        # ----------------------------------------------------

        input_data = {

            "age": age,

            "gender": gender_encoded,

            "bmi": bmi,

            "diabetes_type": diabetes_encoded,

            "meal_time": meal_encoded,

            "carb_intake": carb_intake,

            "protein_intake": protein_intake,

            "fat_intake": fat_intake,

            "fiber_intake": fiber_intake,

            "sugar_intake": sugar_intake,

            "glycemic_index": glycemic_index,

            "portion_size": portion_size,

            "water_intake": water_intake,

            "insulin_dose": insulin_dose,

            "medication_adherence": medication_encoded,

            "physical_activity": physical_activity,

            "stress_level": stress_level,

            "sleep_hours": sleep_hours,

            "smoking_status": smoking_encoded,

            "alcohol_consumption": alcohol_encoded,

            "pre_meal_glucose": pre_meal_glucose,

            "glycemic_load": glycemic_load,

            "carb_fiber_ratio": carb_fiber_ratio,

            "high_carb_flag": high_carb_flag,

            "high_gl_flag": high_gl_flag,

            "low_activity_flag": low_activity_flag,

            "sugar_carb_ratio": sugar_carb_ratio,

            "bmi_category": bmi_category,

            "insulin_carb_ratio": insulin_carb_ratio,

            "poor_sleep_flag": poor_sleep_flag,

            "high_stress_flag": high_stress_flag
        }


        # Convert to dataframe
        input_df = pd.DataFrame([input_data])


        # Exact training feature order
        input_df = input_df[feature_order]


        # ----------------------------------------------------
        # APPLY TRAINING SCALER
        # ----------------------------------------------------

        scaled_df = input_df.copy()

        scaled_df[scale_cols] = scaler.transform(
            input_df[scale_cols]
        )

        return input_df, scaled_df


    # ========================================================
    # PREDICTION
    # ========================================================

    st.divider()

    if st.button(
        "Run Model Prediction",
        type="primary",
        use_container_width=True
    ):

        raw_input, model_input = prepare_input()

        probability = model.predict_proba(
            model_input
        )[0][1]

        prediction = model.predict(
            model_input
        )[0]

        probability_pct = probability * 100


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.header("Model output")

        metric1, metric2 = st.columns(2)

        metric1.metric(
            "Predicted spike probability",
            f"{probability_pct:.1f}%"
        )

        predicted_class = (
            "Spike"
            if prediction == 1
            else "No Spike"
        )

        metric2.metric(
            "Predicted class",
            predicted_class
        )


        if prediction == 1:

            st.warning(
                "For these hypothetical inputs, the model classified "
                "the example as a glucose-spike event."
            )

        else:

            st.success(
                "For these hypothetical inputs, the model classified "
                "the example as a non-spike event."
            )


        # ----------------------------------------------------
        # MODEL CONTEXT
        # ----------------------------------------------------

        st.subheader("Derived model features")

        st.write(
            "The application automatically generated several "
            "features used by the model from the values entered above."
        )

        context_col1, context_col2 = st.columns(2)

        with context_col1:

            st.metric(
                "Calculated glycaemic load",
                f"{raw_input['glycemic_load'].iloc[0]:.1f}"
            )

            st.metric(
                "Carbohydrate-to-fibre ratio",
                f"{raw_input['carb_fiber_ratio'].iloc[0]:.2f}"
            )

        with context_col2:

            st.metric(
                "Sugar-to-carbohydrate ratio",
                f"{raw_input['sugar_carb_ratio'].iloc[0]:.2f}"
            )

            st.metric(
                "Insulin-to-carbohydrate ratio",
                f"{raw_input['insulin_carb_ratio'].iloc[0]:.3f}"
            )


        st.caption(
            "These values are shown to explain how the application "
            "transforms the hypothetical inputs. They are not medical "
            "recommendations or clinical thresholds."
        )


        # ----------------------------------------------------
        # IMPORTANT DISCLAIMER
        # ----------------------------------------------------

        st.divider()

        st.warning(
            "This prediction is an output from a portfolio machine-learning "
            "model trained on the project's dataset. It has not been "
            "clinically validated and must not be used to make healthcare, "
            "medication, insulin-dosing or dietary decisions."
        )


# ============================================================
# ABOUT TAB
# ============================================================

with about_tab:

    st.header("About this project")

    st.markdown(
        """
        **NutriGlyc AI** is a machine-learning project exploring whether
        dietary, clinical and lifestyle variables can be used to classify
        post-meal glucose-spike events.

        The project was developed as part of a **data science internship**
        and demonstrates an end-to-end workflow covering data-quality
        assessment, exploratory analysis, feature engineering, model
        comparison, hyperparameter tuning, explainability and deployment.

        ### The modelling problem

        The original cleaned dataset contained **5,000 records** covering
        demographic, dietary, clinical and lifestyle characteristics.

        Two variables, `post_meal_glucose` and `glucose_change`, were
        excluded before modelling because they contain information available
        only after the outcome and would therefore introduce **data leakage**.

        Logistic Regression, Random Forest and XGBoost were compared.

        The original tuned XGBoost model achieved:

        - **Accuracy: 77.6%**
        - **Recall: 83.15%**
        - **F1 Score: 0.7746**
        - **ROC-AUC: 0.8531**

        Logistic Regression produced a marginally higher ROC-AUC of
        **0.8540**, while tuned XGBoost achieved the strongest recall and
        F1 score.

        ### Deployment model

        The original modelling pipeline contained **32 features**.

        One feature, `meal_risk_score`, originated from the supplied
        dataset but could not be reliably reconstructed from new raw
        inputs. Rather than inventing a calculation for deployment, I
        retrained the application model without this feature.

        The resulting **31-feature deployment model** achieved:

        - **Accuracy: 76.6%**
        - **Precision: 71.81%**
        - **Recall: 81.43%**
        - **F1 Score: 0.7632**
        - **ROC-AUC: 0.8528**

        This retained almost all of the original model's discriminatory
        performance while allowing every application feature to be
        reproduced from the entered values.

        ### Explainability

        SHAP analysis of the original tuned XGBoost model showed that the
        most influential features included:

        - Carbohydrate intake
        - Insulin-to-carbohydrate ratio
        - Glycaemic load
        - Physical activity
        - Stress level

        These are **model relationships within the project dataset**.
        Feature importance does not establish causation and should not be
        interpreted as clinical evidence.

        ### Purpose of this application

        The Prediction Explorer demonstrates how the trained model responds
        to different **hypothetical input combinations**.

        It is intended to demonstrate machine-learning deployment and
        feature engineering, not to assess an individual's health.
        """
    )


    # --------------------------------------------------------
    # MODEL PERFORMANCE CARDS
    # --------------------------------------------------------

    st.divider()

    st.subheader("Deployment model performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        "76.6%"
    )

    col2.metric(
        "Recall",
        "81.43%"
    )

    col3.metric(
        "F1 Score",
        "0.763"
    )

    col4.metric(
        "ROC-AUC",
        "0.853"
    )


    # --------------------------------------------------------
    # TECHNICAL SUMMARY
    # --------------------------------------------------------

    st.subheader("Technical approach")

    st.markdown(
        """
        **Model:** Tuned XGBoost classifier  
        **Features:** 31 reproducible model inputs  
        **Train/test split:** 80/20 stratified  
        **Tuning:** RandomizedSearchCV with 5-fold cross-validation  
        **Preprocessing:** StandardScaler fitted on training data only  
        **Explainability:** SHAP analysis  
        **Deployment:** Streamlit
        """
    )


    st.divider()

    st.warning(
        "NutriGlyc AI is an educational data-science portfolio project. "
        "The model has not been clinically validated and this application "
        "is not a medical device. Outputs must not be used for diagnosis, "
        "treatment, medication, insulin dosing, dietary planning or other "
        "healthcare decisions."
    )