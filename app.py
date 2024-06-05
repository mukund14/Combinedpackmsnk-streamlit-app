import streamlit as st
import pandas as pd
from combinedpackmsnk import func

def main():
    st.title("combinedpackmsnk: Data Analysis and Machine Learning")

    st.sidebar.title("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    header_row = st.sidebar.number_input("Header Row Number", min_value=0, value=0, step=1)

    if uploaded_file is not None:
        # Read the uploaded file as a pandas DataFrame
        df = pd.read_csv(uploaded_file, header=header_row)
        st.write("Data Preview:")
        st.write(df.head())

        st.sidebar.subheader("Data Preprocessing")
        preprocess = st.sidebar.checkbox("Preprocess Data")

        if preprocess:
            st.sidebar.write("Data will be preprocessed...")

        st.sidebar.subheader("Choose Analysis")
        analysis = st.sidebar.selectbox(
            "Select Analysis Type",
            ["None", "Statistical Analysis", "Machine Learning", "Clustering", "PCA"]
        )

        if analysis == "Statistical Analysis":
            st.subheader("Statistical Analysis")
            st.write(df.describe())

        if analysis == "Machine Learning":
            ml_task = st.sidebar.radio("Select Task", ["Classification", "Regression"])
            if ml_task == "Classification":
                model = st.sidebar.selectbox(
                    "Select Model",
                    ["Random Forest", "Logistic Regression", "Support Vector Machine", "Gradient Boosting", "AdaBoost", "XGBoost"]
                )
            elif ml_task == "Regression":
                model = st.sidebar.selectbox(
                    "Select Model",
                    ["Random Forest", "Linear Regression", "Support Vector Machine", "Gradient Boosting", "AdaBoost", "XGBoost"]
                )

            target = st.sidebar.selectbox("Select Target Variable", df.columns)
            st.write(f"Selected Model: {model}")
            st.write(f"Target Variable: {target}")

            if st.sidebar.button("Run Analysis"):
                # Call the function from the package with the DataFrame directly
                results = func(df)
                st.write("Results:")
                st.write(results)

        if analysis == "Clustering":
            clustering_model = st.sidebar.selectbox("Select Clustering Model", ["KMeans", "DBSCAN"])
            st.write(f"Selected Clustering Model: {clustering_model}")

            if st.sidebar.button("Run Clustering"):
                results = func(df)
                st.write("Results:")
                st.write(results)

        if analysis == "PCA":
            n_components = st.sidebar.number_input("Number of Components for PCA", min_value=1, max_value=min(len(df.columns), 10), value=2)
            st.write(f"Number of Components for PCA: {n_components}")

            if st.sidebar.button("Run PCA"):
                results = func(df)
                st.write("Results:")
                st.write(results)

if __name__ == "__main__":
    main()
