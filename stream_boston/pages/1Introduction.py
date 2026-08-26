import streamlit as st

st.set_page_config(
    page_title="Introduction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Boston Housing Price Prediction System")

st.markdown("---")

st.header("📌 Project Overview")

st.write("""
Boston Housing Price Prediction is a Machine Learning regression project
that predicts the value of houses based on different numerical features
related to the Boston housing dataset.

The prediction is based on housing-related information such as:

- Industrial proportion
- Charles River accessibility
- Nitric oxide concentration
- Average number of rooms
- Age of houses
- Distance to employment centres
- Accessibility to radial highways
- Property-tax rate
- Pupil-teacher ratio
- Lower status of the population
""")

st.markdown("---")

st.header("🎯 Objective")

st.write("""
The objective of this project is to analyze the Boston Housing dataset
and develop Machine Learning regression models that can predict the
median value of a house based on its input features.

The project also provides an interactive Streamlit interface where
users can enter their own housing-related values and obtain a predicted
house value.
""")

st.markdown("---")

st.header("📂 Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Dataset Name**

    Boston Housing Dataset
    """)

with col2:
    st.info("""
    **Target Variable**

    MEDV
    """)

st.markdown("---")

st.header("📊 Problem Type")

st.success("""
✔ Supervised Machine Learning

✔ Regression Problem

✔ Numerical Dataset

✔ Continuous Target Prediction
""")

st.markdown("---")

st.header("🤖 Machine Learning Algorithms Used")

st.success("✔ Regression Model 1")
st.success("✔ Regression Model 2")
st.success("✔ Regression Model 3")
st.success("✔ Regression Model 4")

st.markdown("---")

st.header("📊 Project Workflow")

st.write("""
1. Data Collection

2. Data Cleaning

3. Exploratory Data Analysis (EDA)

4. Data Preprocessing

5. Feature Selection

6. Model Training

7. Model Evaluation

8. Model Comparison

9. House Value Prediction
""")

st.markdown("---")

st.header("🛠 Technologies Used")

st.write("""
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit
""")

st.markdown("---")

st.header("🎯 Application Features")

st.write("""
- 📊 Explore the dataset through EDA
- 📈 Analyze numerical features
- 🔍 Understand relationships between variables
- 🤖 Use trained regression models
- 🏠 Enter your own house-related values
- 🔮 Predict the estimated house value
""")

st.markdown("---")

st.success("✅ Use the sidebar to open the EDA and Prediction pages.")