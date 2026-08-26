import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# BOSTON HOUSE PRICE PREDICTION
# ============================================================

st.set_page_config(
    page_title="Boston House Price Prediction",
    page_icon="🏠",
    layout="wide",
)

# Project structure:
# Boston prediction/
# ├── models/
# │   ├── Linear_Model.pkl
# │   ├── Linear_Metrics.pkl
# │   ├── DT_Model.pkl
# │   ├── DT_Metrics.pkl
# │   ├── RF_Model.pkl
# │   ├── RF_Metrics.pkl
# │   ├── SVM_Model.pkl
# │   ├── SVM_Metrics.pkl
# │   ├── KNN_Model.pkl
# │   ├── KNN_Metrics.pkl
# │   ├── GB_Model.pkl
# │   ├── GB_Metrics.pkl
# │   ├── AB_Model.pkl
# │   └── AB_Metrics.pkl
# └── pages/
#     └── Prediction.py

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# These are exactly the 11 input columns used in the notebook.
# Target column is "medv", so it is NOT requested from the user.
FEATURES = [
    "ID",
    "indus",
    "chas",
    "nox",
    "rm",
    "age",
    "dis",
    "rad",
    "tax",
    "ptratio",
    "lstat",
]

MODELS = {
    "Linear Regression": ("Linear_Model.pkl", "Linear_Metrics.pkl"),
    "Decision Tree": ("DT_Model.pkl", "DT_Metrics.pkl"),
    "Random Forest": ("RF_Model.pkl", "RF_Metrics.pkl"),
    "SVM": ("SVM_Model.pkl", "SVM_Metrics.pkl"),
    "KNN": ("KNN_Model.pkl", "KNN_Metrics.pkl"),
    "Gradient Boosting": ("GB_Model.pkl", "GB_Metrics.pkl"),
    "AdaBoost": ("AB_Model.pkl", "AB_Metrics.pkl"),
}


# ============================================================
# LOAD FUNCTIONS
# ============================================================

@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_metrics(path):
    metrics = joblib.load(path)

    return {
        "MAE": float(metrics["MAE"]),
        "MSE": float(metrics["MSE"]),
        "RMSE": float(metrics["RMSE"]),
        "R2": float(metrics["R2_Score"]),
    }


@st.cache_data
def load_dataset():
    possible_files = [
        BASE_DIR / "Clean_Data_Boston.csv",
        BASE_DIR / "clean_data_boston.csv",
        BASE_DIR / "boston_data.csv",
        BASE_DIR / "boston_data.csv",
    ]

    for file in possible_files:
        if file.exists():
            return pd.read_csv(file)

    return None


def available_models():
    result = {}

    for name, (model_file, metric_file) in MODELS.items():
        model_path = MODEL_DIR / model_file
        metric_path = MODEL_DIR / metric_file

        if model_path.exists():
            result[name] = {
                "model": model_path,
                "metrics": metric_path if metric_path.exists() else None,
            }

    return result


def format_money(value):
    return f"${value:,.2f}"


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #667085;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .prediction-card {
        padding: 1.5rem;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,.25);
        text-align: center;
        margin-top: 1rem;
    }

    .prediction-label {
        font-size: 1rem;
        opacity: .7;
    }

    .prediction-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: .4rem 0;
    }

    .winner {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: rgba(46, 204, 113, .12);
        border: 1px solid rgba(46, 204, 113, .35);
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏠 Boston House Price Prediction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Enter the 11 property features, select a machine-learning algorithm,
    and predict the Boston house value. The same property is also evaluated
    by all available models so you can compare their predictions and
    performance.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CHECK MODELS
# ============================================================

available = available_models()

if not available:
    st.error("No trained model was found.")
    st.code(f"Expected model folder:\n{MODEL_DIR}")
    st.stop()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("🏆 Model Performance")

metric_rows = []

for name, files in available.items():

    if files["metrics"] is None:
        continue

    try:
        m = load_metrics(files["metrics"])

        metric_rows.append(
            {
                "Model": name,
                "MAE": m["MAE"],
                "MSE": m["MSE"],
                "RMSE": m["RMSE"],
                "R²": m["R2"],
            }
        )

    except Exception:
        pass


metrics_df = pd.DataFrame(metric_rows)

if not metrics_df.empty:

    # Higher R² is better.
    best_r2_row = metrics_df.loc[metrics_df["R²"].idxmax()]

    # Lower RMSE is better.
    best_rmse_row = metrics_df.loc[metrics_df["RMSE"].idxmin()]

    # Lower MAE is better.
    best_mae_row = metrics_df.loc[metrics_df["MAE"].idxmin()]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🥇 Highest R²",
            best_r2_row["Model"],
            f"{best_r2_row['R²']:.4f}",
        )

    with col2:
        st.metric(
            "🎯 Lowest RMSE",
            best_rmse_row["Model"],
            f"{best_rmse_row['RMSE']:.4f}",
        )

    with col3:
        st.metric(
            "📉 Lowest MAE",
            best_mae_row["Model"],
            f"{best_mae_row['MAE']:.4f}",
        )

    # The notebook's evaluation shows Gradient Boosting as the
    # strongest model on R², RMSE and MAE.
    if (
        best_r2_row["Model"]
        == best_rmse_row["Model"]
        == best_mae_row["Model"]
    ):
        st.markdown(
            f"""
            <div class="winner">
            🏆 <b>Recommended Model: {best_r2_row["Model"]}</b><br>
            This model has the highest R² and the lowest RMSE and MAE
            among the models with available metric files.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.dataframe(
        metrics_df.sort_values("R²", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

else:
    st.warning(
        "Model files were found, but no readable metric files were found."
    )


# ============================================================
# MODEL SELECTION
# ============================================================

st.divider()

st.header("🤖 Select Algorithm")

model_names = list(available.keys())

recommended_model = (
    best_r2_row["Model"]
    if not metrics_df.empty
    else model_names[0]
)

selected_model = st.selectbox(
    "Choose the algorithm for your main prediction:",
    model_names,
    index=model_names.index(recommended_model),
)

st.caption(
    f"Selected algorithm: **{selected_model}**"
)


# ============================================================
# LOAD DATA FOR INPUT RANGES
# ============================================================

df = load_dataset()

# Safe defaults based on the Boston dataset used by the project.
DEFAULTS = {
    "ID": (1.0, 506.0, 253.0),
    "indus": (0.0, 30.0, 10.0),
    "chas": (0.0, 1.0, 0.0),
    "nox": (0.3, 1.0, 0.55),
    "rm": (1.0, 10.0, 6.2),
    "age": (0.0, 100.0, 77.0),
    "dis": (1.0, 15.0, 3.2),
    "rad": (1.0, 24.0, 5.0),
    "tax": (100.0, 800.0, 330.0),
    "ptratio": (10.0, 30.0, 18.0),
    "lstat": (0.0, 50.0, 12.0),
}


def get_limits(feature):

    if df is not None and feature in df.columns:

        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        ).dropna()

        if not values.empty:

            return (
                float(values.min()),
                float(values.max()),
                float(values.median()),
            )

    return DEFAULTS[feature]


def create_input():

    values = {}

    st.subheader("🏡 Enter Property Details")

    st.caption(
        "The target variable **medv** is not entered because it is the "
        "value the model predicts."
    )

    with st.form("prediction_form"):

        columns = st.columns(3)

        for i, feature in enumerate(FEATURES):

            with columns[i % 3]:

                low, high, default = get_limits(feature)

                if feature == "chas":

                    values[feature] = st.selectbox(
                        "chas — Charles River",
                        [0, 1],
                        index=int(default),
                        help="0 = not bounded by Charles River, 1 = bounded.",
                    )

                elif feature in ["ID", "rad", "tax"]:

                    values[feature] = st.number_input(
                        feature,
                        min_value=int(low),
                        max_value=int(high),
                        value=int(round(default)),
                        step=1,
                    )

                else:

                    values[feature] = st.number_input(
                        feature,
                        min_value=float(low),
                        max_value=float(high),
                        value=float(default),
                        step=0.01,
                        format="%.4f",
                    )

        predict = st.form_submit_button(
            "🚀 Predict House Price",
            use_container_width=True,
            type="primary",
        )

    return values, predict


input_values, predict_clicked = create_input()


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    input_df = pd.DataFrame(
        [input_values],
        columns=FEATURES,
    )

    st.divider()
    st.header("💰 Prediction Result")

    predictions = {}
    errors = {}

    # Predict the SAME user input using every available model.
    for name, files in available.items():

        try:

            model = load_model(files["model"])

            prediction = model.predict(input_df)

            predictions[name] = float(prediction[0])

        except Exception as error:

            errors[name] = str(error)

    # Main selected-model prediction.
    if selected_model not in predictions:

        st.error(
            f"{selected_model} could not make a prediction for this input."
        )

        if selected_model in errors:
            st.code(errors[selected_model])

        st.stop()

    selected_prediction = predictions[selected_model]

    # ========================================================
    # SELECTED MODEL METRICS
    # ========================================================

    st.subheader(f"📊 {selected_model} Test Metrics")

    selected_metric_file = available[selected_model]["metrics"]

    if selected_metric_file is not None:

        selected_metrics = load_metrics(
            selected_metric_file
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "MAE",
            f"{selected_metrics['MAE']:.4f}",
        )

        m2.metric(
            "MSE",
            f"{selected_metrics['MSE']:.4f}",
        )

        m3.metric(
            "RMSE",
            f"{selected_metrics['RMSE']:.4f}",
        )

        m4.metric(
            "R²",
            f"{selected_metrics['R2']:.4f}",
        )

    # ========================================================
    # ALL MODEL PREDICTIONS
    # ========================================================

    st.subheader("🔬 Same Property — All Model Predictions")

    prediction_rows = []

    for name in model_names:

        prediction_rows.append(
            {
                "Model": name,
                "Predicted MEDV": predictions.get(name),
                "MAE": (
                    metrics_df.loc[
                        metrics_df["Model"] == name,
                        "MAE",
                    ].iloc[0]
                    if not metrics_df[
                        metrics_df["Model"] == name
                    ].empty
                    else None
                ),
                "RMSE": (
                    metrics_df.loc[
                        metrics_df["Model"] == name,
                        "RMSE",
                    ].iloc[0]
                    if not metrics_df[
                        metrics_df["Model"] == name
                    ].empty
                    else None
                ),
                "R²": (
                    metrics_df.loc[
                        metrics_df["Model"] == name,
                        "R²",
                    ].iloc[0]
                    if not metrics_df[
                        metrics_df["Model"] == name
                    ].empty
                    else None
                ),
            }
        )

    prediction_table = pd.DataFrame(prediction_rows)

    st.dataframe(
        prediction_table.style.format(
            {
                "Predicted MEDV": "{:.2f}",
                "MAE": "{:.4f}",
                "RMSE": "{:.4f}",
                "R²": "{:.4f}",
            },
            na_rep="N/A",
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    if not metrics_df.empty:

        st.subheader(
            f"⚖️ Why {selected_model} Is Better or Worse"
        )

        selected_metric_rows = metrics_df[
            metrics_df["Model"] == selected_model
        ]

        if not selected_metric_rows.empty:

            selected_row = selected_metric_rows.iloc[0]

            comparison_rows = []

            for name in metrics_df["Model"]:

                if name == selected_model:
                    continue

                other = metrics_df[
                    metrics_df["Model"] == name
                ].iloc[0]

                # Positive R² difference means selected is better.
                r2_difference = (
                    selected_row["R²"] - other["R²"]
                )

                # Positive advantage means selected has lower error.
                rmse_advantage = (
                    other["RMSE"] - selected_row["RMSE"]
                )

                mae_advantage = (
                    other["MAE"] - selected_row["MAE"]
                )

                better_count = sum(
                    [
                        r2_difference > 0,
                        rmse_advantage > 0,
                        mae_advantage > 0,
                    ]
                )

                if better_count == 3:
                    verdict = "🏆 Better on all 3"
                elif better_count == 2:
                    verdict = "✅ Better on 2 of 3"
                elif better_count == 1:
                    verdict = "⚠️ Better on 1 of 3"
                else:
                    verdict = "❌ Other model is better"

                comparison_rows.append(
                    {
                        "Compared With": name,
                        "Verdict": verdict,
                        "R² Difference": r2_difference,
                        "RMSE Advantage": rmse_advantage,
                        "MAE Advantage": mae_advantage,
                    }
                )

            comparison_df = pd.DataFrame(comparison_rows)

            st.dataframe(
                comparison_df.style.format(
                    {
                        "R² Difference": "{:+.4f}",
                        "RMSE Advantage": "{:+.4f}",
                        "MAE Advantage": "{:+.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            # Human-readable conclusion.
            best_model = best_r2_row["Model"]

            if selected_model == best_model:

                st.success(
                    f"🏆 **{selected_model} is the recommended model "
                    f"according to the saved evaluation metrics.** "
                    f"It has the highest R² among the available models."
                )

            else:

                st.warning(
                    f"**{selected_model}** is your selected model, but "
                    f"**{best_model}** has the highest R² in the saved "
                    f"evaluation metrics."
                )

    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    with st.expander("🧾 View Input Sent to the Models"):

        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # PREDICTION DIFFERENCE CHART
    # ========================================================

    if len(predictions) > 1:

        st.subheader("📈 Prediction Comparison")

        chart = pd.DataFrame(
            {
                "Model": list(predictions.keys()),
                "Predicted MEDV": list(predictions.values()),
            }
        ).set_index("Model")

        st.bar_chart(
            chart,
            use_container_width=True,
        )


# ============================================================
# METRIC EXPLANATION
# ============================================================

st.divider()

with st.expander("ℹ️ Understand the Evaluation Metrics"):

    st.markdown(
        """
        **R² (R-squared) — higher is better**

        Measures how well the model explains variation in house prices.

        **MAE (Mean Absolute Error) — lower is better**

        Average absolute difference between actual and predicted prices.

        **MSE (Mean Squared Error) — lower is better**

        Squares prediction errors, giving larger errors more weight.

        **RMSE (Root Mean Squared Error) — lower is better**

        The square root of MSE and expressed on the target's scale.

        **Important:** The MAE, MSE, RMSE and R² shown above are the
        saved test-set evaluation metrics from your trained models.
        They are not recalculated for the new property because the actual
        MEDV of that new property is unknown.
        """
    )
