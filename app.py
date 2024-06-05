import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from combinedpackmsnk import func

def preprocess_data(df, target, id_col, scaling_option):
    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col].astype(str))

    # Handle missing values
    df = df.fillna(df.mean())  # Fill numeric columns with mean
    df = df.fillna(df.mode().iloc[0])  # Fill categorical columns with mode

    # Apply scaling
    if scaling_option == "StandardScaler":
        scaler = StandardScaler()
    elif scaling_option == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = None

    if scaler:
        features = df.drop(columns=[target, id_col]) if id_col else df.drop(columns=[target])
        scaled_features = scaler.fit_transform(features)
        df[features.columns] = scaled_features

    return df

def main():
    st.title("combinedpackmsnk: Data Analysis and Machine Learning")

    st.sidebar.title("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    header_row = st.sidebar.number_input("Header Row Number", min_value=0, value=0, step=1)

    if uploaded_file is not None:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file, header=header_row)

        # Display the first 5 rows of the dataset
        st.write("First 5 rows of the dataset:")
        st.write(df.head(5))

        # Ask for target variable and ID column
        target = st.sidebar.selectbox("Select Target Variable", df.columns)
        id_col = st.sidebar.text_input("ID Column (leave blank if none)", "")

        st.sidebar.subheader("Data Preprocessing")
        preprocess = st.sidebar.checkbox("Preprocess Data")

        scaling_option = "None"
        if preprocess:
            scaling_option = st.sidebar.radio("Choose a scaler", ["StandardScaler", "MinMaxScaler", "None"])

        st.sidebar.subheader("Choose Analysis")
        analysis = st.sidebar.selectbox(
            "Select Analysis Type",
            ["None", "Statistical Analysis", "Machine Learning", "Clustering", "PCA"]
        )

        ml_task = None
        model = None
        clustering_model = None
        n_components = None

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

        if analysis == "Clustering":
            clustering_model = st.sidebar.selectbox("Select Clustering Model", ["KMeans", "DBSCAN"])

        if analysis == "PCA":
            n_components = st.sidebar.number_input("Number of Components for PCA", min_value=1, max_value=10, value=2)

        if st.sidebar.button("Run Analysis"):
            st.write("Running analysis...")

            # Preprocess the dataframe as needed before passing it to the func function
            if preprocess:
                df = preprocess_data(df, target, id_col, scaling_option)

            # Save the preprocessed file temporarily
            df.to_csv("temp_uploaded_file.csv", index=False)

            # Call the function from the package with the necessary parameters
            try:
                results = func("temp_uploaded_file.csv", header_row_number=header_row)
                st.write("Results:")
                st.write(results)
            except TypeError as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
