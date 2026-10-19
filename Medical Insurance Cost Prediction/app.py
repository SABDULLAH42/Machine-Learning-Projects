import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title= "Insurance Cost Estimator",
    layout= "wide"
)

MODEL_PATH = "insurance_model.pkl"
FEATURES_PATH = "model_features.pkl"

REF_AVG_NON_SMOKER = 8400
REF_AVG_SMOKER  = 32000

@st.cache_resource  # we dont want to load this again and again when our app is running
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    return model, feature_cols

def bmi_category(bmi):
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal"
    elif bmi < 30:
        return "overweight"
    else:
        return "obese"

def build_input_row(age, sex, bmi, children, smoker, region, feature_cols):
    category = bmi_category(bmi)

    row = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
        "bmi_category_obese": 1 if category == "obese" else 0,
        "bmi_category_overweight": 1 if category == "over_weight" else 0,
        "bmi_category_underweight": 1 if category == "under_weight" else 0
    }

    encoded = pd.DataFrame([row])

    return encoded.reindex(columns= feature_cols, fill_value= 0)

model, feature_cols = load_artifacts()

st.title(" 🩸Insurance Cost Estimator ")

if model is None:
    st.error("Model files not found")
    st.stop()

left, right = st.columns([1, 1.2], gap= "large")

with left:
    st.subheader("Patient Details")

    age = st.slider("Age", 18, 64, 30)
    # min=18, max=64, default=30
    sex = st.radio("Sex", ["male", "female"], horizontal=True)
    bmi = st.slider("BMI", 15.0, 55.0, 26.0, 0.1)
    children = st.slider("Children", 0, 5, 0)
    smoker = st.radio("Smoker", ["no", "yes"], horizontal=True)
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    predict_clicked = st.button("Estimate charges", type = "primary", use_container_width= True)

with right:

    st.subheader("Estimated Charges")

    if predict_clicked:
        input_row = build_input_row(age, sex, bmi, children, smoker, region, feature_cols)
        log_pred = model.predict(input_row)[0]
        pred_charges = np.expm1(log_pred)

        st.metric("Predicted annual charge", f"${pred_charges:,.2f}")

        st.divider()

        compare_df = pd.DataFrame({
            "Group" : ["Your Prediction", "Non-smoker avg.", "Smoker avg."],
            "Charges" : [pred_charges, REF_AVG_NON_SMOKER, REF_AVG_SMOKER]
        }).set_index("Group")

        st.bar_chart(compare_df)

    else:
        st.info("Fill in the details and click **Estimate charges**.")



