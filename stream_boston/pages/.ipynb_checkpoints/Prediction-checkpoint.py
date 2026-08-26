import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Boston Housing Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Boston Housing Price Prediction")

st.markdown("---")

# ======================================
# Load Models
# ======================================

MODELS = {
    "Linear Regression": "models/LinearRegression.pkl",
    "Decision Tree": "models/DecisionTree.pkl",
    "Random Forest": "models/RandomForest.pkl",
    "K-Nearest Neighbors": "models/KNNRegression.pkl"
}

model_name = st.selectbox(
    "Select Regression Model",
    list(MODELS.keys())
)

model = joblib.load(MODELS[model_name])

st.success(f"✅ {model_name} Loaded Successfully")

st.markdown("---")

st.header("🏡 Enter House Details")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        indus = st.number_input(
            "INDUS (Non-retail business acres)",
            min_value=0.0,
            value=11.0
        )

        chas = st.selectbox(
            "CHAS (Charles River)",
            [0,1],
            format_func=lambda x: "Yes" if x==1 else "No"
        )

        nox = st.number_input(
            "NOX (Nitric Oxide Concentration)",
            min_value=0.0,
            value=0.55
        )

        rm = st.number_input(
            "RM (Average Rooms)",
            min_value=1.0,
            value=6.2
        )

        age = st.number_input(
            "AGE (% houses built before 1940)",
            min_value=0.0,
            max_value=100.0,
            value=65.0
        )

    with col2:

        dis = st.number_input(
            "DIS (Distance to Employment Centres)",
            min_value=0.0,
            value=4.5
        )

        rad = st.number_input(
            "RAD (Accessibility to Highways)",
            min_value=1,
            value=5
        )

        tax = st.number_input(
            "TAX (Property Tax Rate)",
            min_value=0,
            value=300
        )

        ptratio = st.number_input(
            "PTRATIO (Pupil-Teacher Ratio)",
            min_value=0.0,
            value=18.0
        )

        lstat = st.number_input(
            "LSTAT (% Lower Status Population)",
            min_value=0.0,
            value=12.0
        )

    predict = st.form_submit_button(
        "🏠 Predict House Price",
        use_container_width=True
    )

# ======================================
# Prediction
# ======================================

if predict:

    input_data = pd.DataFrame({

        "indus":[indus],
        "chas":[chas],
        "nox":[nox],
        "rm":[rm],
        "age":[age],
        "dis":[dis],
        "rad":[rad],
        "tax":[tax],
        "ptratio":[ptratio],
        "lstat":[lstat]

    })

    st.subheader("Input Data")

    st.dataframe(input_data,use_container_width=True)

    prediction = model.predict(input_data)[0]

    st.markdown("---")

    st.header("🏡 Predicted House Price")

    st.success(f"Estimated MEDV : **${prediction:.2f} (in $1000s)**")

    st.info(
        f"Estimated House Price = **${prediction*1000:,.0f}**"
    )

    st.markdown("---")

    st.subheader("Prediction Summary")

    summary = pd.DataFrame({
        "Feature":input_data.columns,
        "Value":input_data.iloc[0].values
    })

    st.dataframe(summary,use_container_width=True)

    st.success("Prediction Completed Successfully ✅")