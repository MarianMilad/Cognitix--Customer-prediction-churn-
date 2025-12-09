import streamlit as st
import pandas as pd
import pickle

# -----------------------
# Page config & title
# -----------------------
st.set_page_config(page_title="Cognitix | Customer Churn Prediction", layout="centered")
st.title("Cognitix — Customer Churn Prediction")

# -----------------------
# Logo (if exists)
# -----------------------
st.image("logo.png", width=150)

st.write("---")

# -----------------------
# Load trained model
# -----------------------
@st.cache_resource
def load_model():
    with open("rf_gscv_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------
# Upload dataset
# -----------------------
uploaded_file = st.file_uploader("Upload your customer CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df)

    st.write("---")
    st.subheader("⚙️ Select features for Prediction")

    # exclude any non-feature columns, or let user select
    feature_cols = st.multiselect("Select feature columns:", df.columns)

    if st.button("Predict Churn"):
        if not feature_cols:
            st.error("Please select features to use for prediction.")
        else:
            try:
                X = df[feature_cols]
                preds = model.predict(X)
                df_result = df.copy()
                df_result["Churn_Prediction"] = preds
                st.success("✅ Prediction done!")
                st.write("### Predictions")
                st.dataframe(df_result)
                csv = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error during prediction: {e}")
