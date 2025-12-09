import streamlit as st
import pandas as pd
import pickle
import joblib
from xgboost import XGBClassifier

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Cognitix | Customer Churn Prediction",
    layout="centered",
)

# -----------------------
# LOGO + HEADER
# -----------------------
st.markdown(
    """
    <div style="text-align: center;">
        <img src="images/cognitix.png" width="180">
        <h2>Customer Churn Prediction App</h2>
        <p>Powered by Cognitix</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

# -----------------------
# LOAD MODEL
# -----------------------
st.subheader("📦 Upload Your Trained Model")
model_file = st.file_uploader(
    "Upload your trained model (.pkl / .joblib / .json)", type=["pkl", "joblib", "json"]
)

model = None
if model_file:
    file_name = model_file.name.lower()
    try:
        if file_name.endswith(".pkl"):
            model = pickle.load(model_file)
            st.success("✅ Model loaded successfully (pickle).")
        elif file_name.endswith(".joblib"):
            model = joblib.load(model_file)
            st.success("✅ Model loaded successfully (joblib).")
        elif file_name.endswith(".json"):
            model = XGBClassifier()
            model.load_model(model_file)
            st.success("✅ XGBoost model loaded successfully (.json).")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")

st.write("---")

# -----------------------
# CSV UPLOAD + PREVIEW
# -----------------------
st.subheader("📁 Upload Customer Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="csv_upload")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df)

        st.write("---")

        # -----------------------
        # SELECT FEATURES + TARGET
        # -----------------------
        st.subheader("⚙️ Select Target & Features")
        target_col = st.selectbox("Select target (Churn) column:", df.columns)
        feature_cols = st.multiselect(
            "Select feature columns:", [col for col in df.columns if col != target_col]
        )

        # -----------------------
        # PREDICT CHURN
        # -----------------------
        if st.button("Predict Churn"):
            if model is None:
                st.error("❌ Please upload your trained model first.")
            else:
                try:
                    X = df[feature_cols]
                    predictions = model.predict(X)
                    df_result = df.copy()
                    df_result["Prediction"] = predictions

                    st.success("✅ Churn prediction completed!")
                    st.write("### Prediction Results")
                    st.dataframe(df_result)

                    # Download predictions as CSV
                    csv_output = df_result.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv_output,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")

    except Exception as e:
        st.error(f"❌ Error reading CSV file: {e}")
