import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Boston EDA",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# PATH CONFIGURATION
# =========================================================

# Current file:
# stream_boston/pages/EDA.py
#
# Project structure from your project:
#
# BOSTON PREDICTION
# │
# ├── boston_data.csv
# │
# └── stream_boston
#     │
#     └── pages
#         └── EDA.py
#
# Therefore:
# pages -> stream_boston -> project root

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BASE_DIR = os.path.dirname(
    CURRENT_DIR
)

PROJECT_DIR = os.path.dirname(
    BASE_DIR
)


# =========================================================
# FIND BOSTON DATASET
# =========================================================

DATA_FILES = [
    os.path.join(PROJECT_DIR, "boston_data.csv"),
    os.path.join(PROJECT_DIR, "clean_data_boston.csv"),
    os.path.join(PROJECT_DIR, "cleaned_data_boston.csv"),
    os.path.join(PROJECT_DIR, "cleaned_data.csv"),
    os.path.join(BASE_DIR, "boston_data.csv"),
    os.path.join(BASE_DIR, "clean_data_boston.csv"),
    os.path.join(BASE_DIR, "cleaned_data_boston.csv"),
    os.path.join(BASE_DIR, "cleaned_data.csv")
]


DATA_PATH = None

for file_path in DATA_FILES:

    if os.path.isfile(file_path):

        DATA_PATH = file_path

        break


# =========================================================
# CHECK DATASET
# =========================================================

if DATA_PATH is None:

    st.error("❌ Boston dataset not found!")

    st.write(
        "The application searched for the dataset in:"
    )

    for path in DATA_FILES:

        st.code(path)

    st.stop()


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data(path):

    return pd.read_csv(path)


try:

    df = load_data(DATA_PATH)

except Exception as e:

    st.error(
        "❌ Could not load the Boston dataset."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# PAGE TITLE
# =========================================================

st.title("📊 Exploratory Data Analysis")

st.write(
    "Explore the Boston Housing dataset through "
    "data inspection, cleaning checks, univariate analysis, "
    "bivariate analysis and multivariate analysis."
)


st.success(
    "Dataset Loaded Successfully ✅"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📊 EDA Navigation")

eda = st.sidebar.radio(
    "EDA Sections",
    [
        "Dataset",
        "Data Cleaning",
        "Univariate Analysis",
        "Bivariate Analysis",
        "Multivariate Analysis"
    ]
)


# =========================================================
# DATASET
# =========================================================

if eda == "Dataset":

    st.header("📋 Dataset Overview")


    # -----------------------------------------------------
    # Dataset Metrics
    # -----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Rows",
            df.shape[0]
        )


    with c2:

        st.metric(
            "Columns",
            df.shape[1]
        )


    with c3:

        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )


    with c4:

        st.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )


    st.markdown("---")


    # -----------------------------------------------------
    # Dataset Location
    # -----------------------------------------------------

    st.subheader("📁 Dataset Information")

    st.write(
        f"**Dataset File:** "
        f"{os.path.basename(DATA_PATH)}"
    )

    st.write(
        f"**Dataset Location:** "
        f"{DATA_PATH}"
    )


    st.markdown("---")


    # -----------------------------------------------------
    # First 10 Rows
    # -----------------------------------------------------

    st.subheader("First 10 Rows")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    st.markdown("---")


    # -----------------------------------------------------
    # Last 10 Rows
    # -----------------------------------------------------

    st.subheader("Last 10 Rows")

    st.dataframe(
        df.tail(10),
        use_container_width=True
    )


    st.markdown("---")


    # -----------------------------------------------------
    # Dataset Information
    # -----------------------------------------------------

    st.subheader("Dataset Information")


    info = pd.DataFrame({

        "Column": df.columns,

        "Data Type":
            df.dtypes.astype(str).values,

        "Missing":
            df.isnull().sum().values,

        "Unique":
            df.nunique().values

    })


    st.dataframe(
        info,
        use_container_width=True
    )


    st.markdown("---")


    # -----------------------------------------------------
    # Statistical Summary
    # -----------------------------------------------------

    st.subheader("Statistical Summary")


    st.dataframe(
        df.describe().T,
        use_container_width=True
    )


    st.markdown("---")


    # -----------------------------------------------------
    # Column Description
    # -----------------------------------------------------

    st.subheader("📚 Feature Description")


    feature_description = pd.DataFrame({

        "Column": [
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
            "medv"
        ],

        "Description": [
            "Unique house ID",
            "Percentage of industrial land",
            "Charles River indicator: 0 = No, 1 = Yes",
            "Nitric oxide concentration",
            "Average number of rooms per dwelling",
            "Percentage of houses built before 1940",
            "Weighted distance to employment centres",
            "Accessibility to radial highways",
            "Property tax rate",
            "Pupil-teacher ratio",
            "Percentage of lower socioeconomic population",
            "Median house value / target"
        ]

    })


    # Only display columns that actually exist
    feature_description = feature_description[
        feature_description["Column"].isin(df.columns)
    ]


    st.dataframe(
        feature_description,
        use_container_width=True
    )


# =========================================================
# DATA CLEANING
# =========================================================

elif eda == "Data Cleaning":

    st.header("🧹 Data Cleaning")


    # -----------------------------------------------------
    # Missing Values
    # -----------------------------------------------------

    st.subheader("Missing Values")


    missing = df.isnull().sum()


    missing_table = pd.DataFrame({

        "Column": missing.index,

        "Missing Values": missing.values

    })


    st.dataframe(
        missing_table,
        use_container_width=True
    )


    total_missing = int(
        df.isnull().sum().sum()
    )


    if total_missing == 0:

        st.success(
            "Total Missing Values: 0 ✅"
        )

    else:

        st.warning(
            f"Total Missing Values: {total_missing}"
        )


    st.markdown("---")


    # -----------------------------------------------------
    # Duplicate Values
    # -----------------------------------------------------

    st.subheader("Duplicate Values")


    duplicate_count = int(
        df.duplicated().sum()
    )


    st.metric(
        "Duplicate Rows",
        duplicate_count
    )


    if duplicate_count == 0:

        st.success(
            "No duplicate rows found ✅"
        )

    else:

        st.warning(
            f"{duplicate_count} duplicate rows found."
        )


    st.markdown("---")


    # -----------------------------------------------------
    # Column Data Types
    # -----------------------------------------------------

    st.subheader("Data Types")


    datatype_table = pd.DataFrame({

        "Column": df.columns,

        "Data Type":
            df.dtypes.astype(str).values

    })


    st.dataframe(
        datatype_table,
        use_container_width=True
    )


    st.markdown("---")


    # -----------------------------------------------------
    # Unique Values
    # -----------------------------------------------------

    st.subheader("Unique Values")


    unique_table = pd.DataFrame({

        "Column": df.columns,

        "Unique Values":
            df.nunique().values

    })


    st.dataframe(
        unique_table,
        use_container_width=True
    )


    st.markdown("---")


    # -----------------------------------------------------
    # Business / Range Validation
    # -----------------------------------------------------

    st.subheader("🔍 Invalid Value Validation")


    validation_results = []


    # CHAS
    if "chas" in df.columns:

        invalid = (
            ~df["chas"].isin([0, 1])
        ).sum()

        validation_results.append(
            [
                "chas",
                "Expected 0 or 1",
                int(invalid)
            ]
        )


    # AGE
    if "age" in df.columns:

        invalid = (
            (df["age"] < 0) |
            (df["age"] > 100)
        ).sum()

        validation_results.append(
            [
                "age",
                "Expected 0 to 100",
                int(invalid)
            ]
        )


    # LSTAT
    if "lstat" in df.columns:

        invalid = (
            (df["lstat"] < 0) |
            (df["lstat"] > 100)
        ).sum()

        validation_results.append(
            [
                "lstat",
                "Expected 0 to 100",
                int(invalid)
            ]
        )


    # RM
    if "rm" in df.columns:

        invalid = (
            df["rm"] <= 0
        ).sum()

        validation_results.append(
            [
                "rm",
                "Must be greater than 0",
                int(invalid)
            ]
        )


    # NOX
    if "nox" in df.columns:

        invalid = (
            df["nox"] <= 0
        ).sum()

        validation_results.append(
            [
                "nox",
                "Must be greater than 0",
                int(invalid)
            ]
        )


    # DIS
    if "dis" in df.columns:

        invalid = (
            df["dis"] <= 0
        ).sum()

        validation_results.append(
            [
                "dis",
                "Must be greater than 0",
                int(invalid)
            ]
        )


    # TAX
    if "tax" in df.columns:

        invalid = (
            df["tax"] <= 0
        ).sum()

        validation_results.append(
            [
                "tax",
                "Must be greater than 0",
                int(invalid)
            ]
        )


    # PTRATIO
    if "ptratio" in df.columns:

        invalid = (
            df["ptratio"] <= 0
        ).sum()

        validation_results.append(
            [
                "ptratio",
                "Must be greater than 0",
                int(invalid)
            ]
        )


    # MEDV
    if "medv" in df.columns:

        invalid = (
            df["medv"] <= 0
        ).sum()

        validation_results.append(
            [
                "medv",
                "Must be greater than 0",
                int(invalid)
            ]
        )


    validation_df = pd.DataFrame(
        validation_results,
        columns=[
            "Column",
            "Validation Rule",
            "Invalid Values"
        ]
    )


    st.dataframe(
        validation_df,
        use_container_width=True
    )


    if len(validation_df) > 0:

        total_invalid = int(
            validation_df["Invalid Values"].sum()
        )


        if total_invalid == 0:

            st.success(
                "No invalid values found according "
                "to the validation rules used in the "
                "Boston EDA notebook ✅"
            )

        else:

            st.warning(
                f"{total_invalid} invalid values found."
            )


    st.markdown("---")


    # -----------------------------------------------------
    # Outlier Boxplots
    # -----------------------------------------------------

    st.subheader("📦 Outlier Boxplots")


    numeric_df = df.select_dtypes(
        include="number"
    )


    fig, axes = plt.subplots(
        nrows=int(
            np.ceil(len(numeric_df.columns) / 3)
        ),
        ncols=3,
        figsize=(15, 8)
    )


    axes = np.array(axes).reshape(-1)


    for index, column in enumerate(
        numeric_df.columns
    ):

        sns.boxplot(
            x=numeric_df[column],
            ax=axes[index]
        )

        axes[index].set_title(
            column
        )


    # Hide unused axes

    for index in range(
        len(numeric_df.columns),
        len(axes)
    ):

        axes[index].axis("off")


    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    st.markdown("---")


    # -----------------------------------------------------
    # IQR Outlier Detection
    # -----------------------------------------------------

    st.subheader(
        "IQR Outlier Detection"
    )


    outlier_results = []


    for column in numeric_df.columns:

        data = numeric_df[column].dropna()


        Q1 = data.quantile(0.25)

        Q3 = data.quantile(0.75)

        IQR = Q3 - Q1


        lower_bound = Q1 - 1.5 * IQR

        upper_bound = Q3 + 1.5 * IQR


        outliers = data[
            (data < lower_bound) |
            (data > upper_bound)
        ]


        outlier_results.append({

            "Column": column,

            "Q1": Q1,

            "Q3": Q3,

            "IQR": IQR,

            "Lower Bound": lower_bound,

            "Upper Bound": upper_bound,

            "Outliers": len(outliers)

        })


    outlier_df = pd.DataFrame(
        outlier_results
    )


    st.dataframe(
        outlier_df,
        use_container_width=True
    )


    st.info(
        "Outlier counts are calculated using the "
        "IQR method used in the Boston EDA notebook. "
        "The notebook also checks whether observed "
        "values are plausible before deciding whether "
        "they should remain unchanged."
    )


# =========================================================
# UNIVARIATE ANALYSIS
# =========================================================

elif eda == "Univariate Analysis":

    st.header("📊 Univariate Analysis")


    # Exact numerical columns used in notebook
    numerical_features = [
        "indus",
        "nox",
        "rm",
        "age",
        "dis",
        "rad",
        "tax",
        "ptratio",
        "lstat",
        "medv"
    ]


    numerical_features = [
        col
        for col in numerical_features
        if col in df.columns
    ]


    analysis_options = [
        "All Numerical Distributions",
        "CHAS",
    ] + numerical_features


    chart = st.selectbox(
        "Select Analysis",
        analysis_options
    )


    # =====================================================
    # ALL NUMERICAL DISTRIBUTIONS
    # =====================================================

    if chart == "All Numerical Distributions":

        st.subheader(
            "Numerical Feature Distributions"
        )


        fig, axes = plt.subplots(
            4,
            3,
            figsize=(15, 12)
        )


        axes = axes.flatten()


        for index, column in enumerate(
            numerical_features
        ):

            sns.histplot(
                data=df,
                x=column,
                kde=True,
                ax=axes[index]
            )


            axes[index].set_title(
                column
            )


        # Hide unused axes

        for index in range(
            len(numerical_features),
            len(axes)
        ):

            axes[index].axis("off")


        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


        st.info(
            "The histograms show the distributions "
            "of the numerical variables included in "
            "the Boston EDA notebook."
        )


    # =====================================================
    # CHAS
    # =====================================================

    elif chart == "CHAS":

        st.subheader(
            "CHAS Distribution"
        )


        c1, c2 = st.columns(2)


        with c1:

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )


            sns.countplot(
                data=df,
                x="chas",
                ax=ax
            )


            for container in ax.containers:

                ax.bar_label(container)


            ax.set_xlabel(
                "CHAS"
            )

            ax.set_ylabel(
                "Count"
            )


            st.pyplot(fig)

            plt.close(fig)


        with c2:

            fig, ax = plt.subplots(
                figsize=(6, 6)
            )


            df["chas"].value_counts().sort_index().plot(
                kind="pie",
                autopct="%1.1f%%",
                ax=ax
            )


            ax.set_ylabel("")


            st.pyplot(fig)

            plt.close(fig)


        st.write(
            "CHAS represents the Charles River indicator."
        )


        st.dataframe(
            df["chas"].value_counts().rename(
                "Count"
            ),
            use_container_width=True
        )


    # =====================================================
    # INDIVIDUAL NUMERICAL FEATURE
    # =====================================================

    elif chart in numerical_features:

        column = chart


        c1, c2 = st.columns(2)


        # -------------------------------------------------
        # Histogram
        # -------------------------------------------------

        with c1:

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )


            sns.histplot(
                data=df,
                x=column,
                kde=True,
                ax=ax
            )


            ax.set_title(
                f"{column} Distribution"
            )


            st.pyplot(fig)

            plt.close(fig)


        # -------------------------------------------------
        # Boxplot
        # -------------------------------------------------

        with c2:

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )


            sns.boxplot(
                x=df[column],
                ax=ax
            )


            ax.set_title(
                f"{column} Boxplot"
            )


            st.pyplot(fig)

            plt.close(fig)


        st.markdown("---")


        # -------------------------------------------------
        # Statistics
        # -------------------------------------------------

        st.subheader(
            f"{column} Statistics"
        )


        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "Minimum",
                f"{df[column].min():.2f}"
            )


        with c2:

            st.metric(
                "Maximum",
                f"{df[column].max():.2f}"
            )


        with c3:

            st.metric(
                "Median",
                f"{df[column].median():.2f}"
            )


        with c4:

            st.metric(
                "Skewness",
                f"{df[column].skew():.2f}"
            )


        st.markdown("---")


        st.subheader(
            f"{column} Statistical Summary"
        )


        st.dataframe(
            df[column].describe().to_frame().T,
            use_container_width=True
        )


# =========================================================
# BIVARIATE ANALYSIS
# =========================================================

elif eda == "Bivariate Analysis":

    st.header("📈 Bivariate Analysis")


    numerical_features = [
        "indus",
        "nox",
        "rm",
        "age",
        "dis",
        "rad",
        "tax",
        "ptratio",
        "lstat"
    ]


    numerical_features = [
        col
        for col in numerical_features
        if col in df.columns
    ]


    options = [
        "Correlation Heatmap",
        "Features vs MEDV",
        "CHAS vs MEDV",
        "CHAS vs RAD"
    ]


    option = st.selectbox(
        "Select Analysis",
        options
    )


    # =====================================================
    # CORRELATION HEATMAP
    # =====================================================

    if option == "Correlation Heatmap":

        st.subheader(
            "Numerical Correlation Heatmap"
        )


        corr = df.corr(
            numeric_only=True
        )


        fig, ax = plt.subplots(
            figsize=(12, 8)
        )


        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )


        st.pyplot(fig)

        plt.close(fig)


    # =====================================================
    # FEATURES VS MEDV
    # =====================================================

    elif option == "Features vs MEDV":

        st.subheader(
            "Numerical Features vs MEDV"
        )


        fig, axes = plt.subplots(
            3,
            3,
            figsize=(15, 10)
        )


        axes = axes.flatten()


        for index, column in enumerate(
            numerical_features
        ):

            sns.scatterplot(
                data=df,
                x=column,
                y="medv",
                ax=axes[index]
            )


            axes[index].set_title(
                f"{column} vs MEDV"
            )


        for index in range(
            len(numerical_features),
            len(axes)
        ):

            axes[index].axis("off")


        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


        st.markdown("---")


        # Correlation with MEDV

        correlation = (
            df[numerical_features + ["medv"]]
            .corr()["medv"]
            .drop("medv")
            .sort_values(
                key=abs,
                ascending=False
            )
        )


        st.subheader(
            "Correlation with MEDV"
        )


        correlation_df = (
            correlation
            .rename("Correlation")
            .to_frame()
        )


        st.dataframe(
            correlation_df,
            use_container_width=True
        )


    # =====================================================
    # CHAS VS MEDV
    # =====================================================

    elif option == "CHAS vs MEDV":

        st.subheader(
            "CHAS vs MEDV"
        )


        c1, c2 = st.columns(2)


        with c1:

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )


            sns.boxplot(
                data=df,
                x="chas",
                y="medv",
                ax=ax
            )


            ax.set_title(
                "CHAS vs MEDV - Boxplot"
            )


            st.pyplot(fig)

            plt.close(fig)


        with c2:

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )


            sns.violinplot(
                data=df,
                x="chas",
                y="medv",
                ax=ax
            )


            ax.set_title(
                "CHAS vs MEDV - Violin Plot"
            )


            st.pyplot(fig)

            plt.close(fig)


        st.markdown("---")


        st.subheader(
            "MEDV Statistics by CHAS"
        )


        chas_summary = (
            df.groupby("chas")["medv"]
            .agg(
                [
                    "count",
                    "mean",
                    "median",
                    "min",
                    "max"
                ]
            )
        )


        st.dataframe(
            chas_summary,
            use_container_width=True
        )


    # =====================================================
    # CHAS VS RAD
    # =====================================================

    elif option == "CHAS vs RAD":

        st.subheader(
            "CHAS vs RAD"
        )


        fig, ax = plt.subplots(
            figsize=(8, 5)
        )


        sns.countplot(
            data=df,
            x="rad",
            hue="chas",
            ax=ax
        )


        ax.set_title(
            "RAD vs CHAS"
        )


        ax.set_xlabel(
            "RAD - Highway Accessibility"
        )


        ax.set_ylabel(
            "Count"
        )


        st.pyplot(fig)

        plt.close(fig)


        st.markdown("---")


        st.subheader(
            "CHAS Distribution by RAD"
        )


        cross_table = pd.crosstab(
            df["rad"],
            df["chas"]
        )


        st.dataframe(
            cross_table,
            use_container_width=True
        )


# =========================================================
# MULTIVARIATE ANALYSIS
# =========================================================

elif eda == "Multivariate Analysis":

    st.header("📊 Multivariate Analysis")


    option = st.selectbox(
        "Select Analysis",
        [
            "Correlation Heatmap",
            "Pairplot",
            "RM - MEDV with CHAS",
            "Outlier Detection",
            "EDA Summary"
        ]
    )


    # =====================================================
    # CORRELATION HEATMAP
    # =====================================================

    if option == "Correlation Heatmap":

        st.subheader(
            "Multivariate Correlation Heatmap"
        )


        corr = df.corr(
            numeric_only=True
        )


        fig, ax = plt.subplots(
            figsize=(12, 8)
        )


        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )


        st.pyplot(fig)

        plt.close(fig)


    # =====================================================
    # PAIRPLOT
    # =====================================================

    elif option == "Pairplot":

        st.subheader(
            "Pairplot"
        )


        pair_columns = [
            "rm",
            "lstat",
            "ptratio",
            "nox",
            "medv"
        ]


        pair_columns = [
            col
            for col in pair_columns
            if col in df.columns
        ]


        pair_fig = sns.pairplot(
            df[pair_columns]
        )


        st.pyplot(
            pair_fig.figure
        )


        plt.close(
            pair_fig.figure
        )


        st.info(
            "The pairplot uses RM, LSTAT, PTRATIO, "
            "NOX and MEDV, matching the Boston EDA notebook."
        )


    # =====================================================
    # RM - MEDV WITH CHAS
    # =====================================================

    elif option == "RM - MEDV with CHAS":

        st.subheader(
            "RM vs MEDV with CHAS"
        )


        fig, ax = plt.subplots(
            figsize=(8, 5)
        )


        sns.scatterplot(
            data=df,
            x="rm",
            y="medv",
            hue="chas",
            ax=ax
        )


        ax.set_title(
            "RM vs MEDV by CHAS"
        )


        ax.set_xlabel(
            "RM - Average Number of Rooms"
        )


        ax.set_ylabel(
            "MEDV - Median House Value"
        )


        st.pyplot(fig)

        plt.close(fig)


    # =====================================================
    # OUTLIER DETECTION
    # =====================================================

    elif option == "Outlier Detection":

        st.subheader(
            "IQR Outlier Detection"
        )


        numeric = df.select_dtypes(
            include="number"
        )


        result = []


        for column in numeric.columns:

            data = numeric[column].dropna()


            Q1 = data.quantile(
                0.25
            )


            Q3 = data.quantile(
                0.75
            )


            IQR = Q3 - Q1


            lower = (
                Q1 -
                1.5 * IQR
            )


            upper = (
                Q3 +
                1.5 * IQR
            )


            outliers = data[
                (data < lower) |
                (data > upper)
            ]


            result.append({

                "Column": column,

                "Q1": Q1,

                "Q3": Q3,

                "IQR": IQR,

                "Lower Bound": lower,

                "Upper Bound": upper,

                "Outliers": len(outliers)

            })


        outlier_df = pd.DataFrame(
            result
        )


        st.dataframe(
            outlier_df,
            use_container_width=True
        )


    # =====================================================
    # EDA SUMMARY
    # =====================================================

    elif option == "EDA Summary":

        st.subheader(
            "📌 EDA Summary"
        )


        st.write(
            """
            The following summary is generated directly
            from the loaded Boston dataset rather than
            using hard-coded observations.
            """
        )


        # -------------------------------------------------
        # Dataset
        # -------------------------------------------------

        st.markdown("### 📋 Dataset")

        st.write(
            f"- **Rows:** {df.shape[0]:,}"
        )

        st.write(
            f"- **Columns:** {df.shape[1]}"
        )

        st.write(
            f"- **Missing values:** "
            f"{df.isnull().sum().sum():,}"
        )

        st.write(
            f"- **Duplicate rows:** "
            f"{df.duplicated().sum():,}"
        )


        # -------------------------------------------------
        # Target
        # -------------------------------------------------

        if "medv" in df.columns:

            st.markdown(
                "### 🎯 Target Variable - MEDV"
            )

            st.write(
                f"- **Minimum MEDV:** "
                f"{df['medv'].min():.2f}"
            )

            st.write(
                f"- **Maximum MEDV:** "
                f"{df['medv'].max():.2f}"
            )

            st.write(
                f"- **Mean MEDV:** "
                f"{df['medv'].mean():.2f}"
            )

            st.write(
                f"- **Median MEDV:** "
                f"{df['medv'].median():.2f}"
            )


        # -------------------------------------------------
        # Strongest correlation with MEDV
        # -------------------------------------------------

        if "medv" in df.columns:

            numeric_columns = (
                df.select_dtypes(
                    include="number"
                )
                .columns
                .tolist()
            )


            feature_columns = [
                col
                for col in numeric_columns
                if col not in ["medv", "ID"]
            ]


            if len(feature_columns) > 0:

                medv_corr = (
                    df[
                        feature_columns +
                        ["medv"]
                    ]
                    .corr()["medv"]
                    .drop("medv")
                )


                strongest_feature = (
                    medv_corr.abs()
                    .idxmax()
                )


                strongest_value = (
                    medv_corr[
                        strongest_feature
                    ]
                )


                st.markdown(
                    "### 🔗 Relationship with MEDV"
                )


                st.write(
                    f"- Feature with the strongest "
                    f"absolute correlation with MEDV "
                    f"in this dataset: **{strongest_feature}**"
                )


                st.write(
                    f"- Correlation value: "
                    f"**{strongest_value:.3f}**"
                )


        # -------------------------------------------------
        # CHAS
        # -------------------------------------------------

        if "chas" in df.columns:

            st.markdown(
                "### 🌊 CHAS"
            )


            chas_counts = (
                df["chas"]
                .value_counts()
                .sort_index()
            )


            for value, count in (
                chas_counts.items()
            ):

                st.write(
                    f"- CHAS = {value}: "
                    f"{count:,} observations"
                )


        # -------------------------------------------------
        # Outliers
        # -------------------------------------------------

        st.markdown(
            "### 📦 IQR Outlier Check"
        )


        numeric = df.select_dtypes(
            include="number"
        )


        total_outliers = 0


        for column in numeric.columns:

            data = numeric[column].dropna()


            Q1 = data.quantile(
                0.25
            )


            Q3 = data.quantile(
                0.75
            )


            IQR = Q3 - Q1


            lower = Q1 - 1.5 * IQR

            upper = Q3 + 1.5 * IQR


            count = (
                (data < lower) |
                (data > upper)
            ).sum()


            total_outliers += int(
                count
            )


        st.write(
            f"- Total IQR outlier observations "
            f"across numerical columns: "
            f"**{total_outliers:,}**"
        )


        st.info(
            "The EDA summary above is calculated from "
            "the current dataset. No fixed Boston-specific "
            "business conclusion has been inserted."
        )


# =========================================================
# DOWNLOAD DATASET
# =========================================================

st.sidebar.markdown("---")


csv = df.to_csv(
    index=False
).encode("utf-8")


st.sidebar.download_button(
    "📥 Download Dataset",
    csv,
    "cleaned_data_boston.csv",
    "text/csv"
)


# =========================================================
# SIDEBAR FOOTER
# =========================================================

st.sidebar.markdown("---")


st.sidebar.success(
    "Boston EDA Completed Successfully ✅"
)