import streamlit as st
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

st.set_page_config(page_title="Cognitix | Customer Churn Prediction", layout="centered")

st.markdown(
    """
    <div style="text-align: center;">
        <img src="images/logo.png" width="180">
        <h2>Customer Churn Prediction App</h2>
        <p>Powered by Cognitix</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("---")

st.subheader("📦 Upload Your Trained Model")
model_file = st.file_uploader(
    "Upload your trained model (.pkl / .joblib / .json)", type=["pkl", "joblib", "json"]
)

model = None
if model_file:
    name = model_file.name.lower()
    try:
        if name.endswith(".pkl"):
            model = pickle.load(model_file)
            st.success("✅ Model loaded (pickle).")
        elif name.endswith(".joblib"):
            model = joblib.load(model_file)
            st.success("✅ Model loaded (joblib).")
        elif name.endswith(".json"):
            xgb = XGBClassifier()
            xgb.load_model(model_file)
            model = xgb
            st.success("✅ XGBoost model loaded (.json).")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")

st.write("---")

st.subheader("📁 Upload Customer Dataset (CSV)")
data_file = st.file_uploader("Upload CSV", type=["csv"], key="data")

if data_file:
    df = pd.read_csv(data_file)
    st.write("### Data Preview")
    st.dataframe(df)

    st.write("---")
    st.subheader("⚙️ Select Target & Features")
    target = st.selectbox("Select target (Churn) column:", df.columns)
    features = st.multiselect("Select feature columns:", [c for c in df.columns if c != target])

    if st.button("Predict Churn"):
        if model is None:
            st.error("❌ You must upload a trained model first.")
        else:
            X = df[features]
            try:
                preds = model.predict(X)
                df["Prediction"] = preds
                st.success("✅ Prediction done!")
                st.write("### Results")
                st.dataframe(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="churn_results.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
