import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Churn Prediction", layout="wide")

# =========================
# CUSTOM UI
# =========================
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>📊 Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
try:
    model = pickle.load(open("model/churn_model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    columns = pickle.load(open("model/columns.pkl", "rb"))
except:
    st.error("❌ Model not found. Run train_model.py first")
    st.stop()

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Go to", [
    "🏠 Home",
    "🔍 Single Prediction",
    "📂 Batch Prediction",
    "📊 Data Visualization"
])

# =========================
# PREPROCESS FUNCTION (FINAL FIX)
# =========================
def preprocess_input(df):
    if isinstance(df, dict):
        df = pd.DataFrame([df])

    df = pd.get_dummies(df)

    # Add missing columns
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only required columns
    df = df[columns]

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Fill missing
    df.fillna(0, inplace=True)

    # Scale
    df = scaler.transform(df)

    return df

# =========================
# HOME PAGE
# =========================
if menu == "🏠 Home":
    st.subheader("Welcome 👋")
    st.write("""
    This is a Machine Learning based Customer Churn Prediction System.

    ### 🚀 Features:
    - 🔍 Single Prediction
    - 📂 Batch Prediction
    - 📊 Data Visualization
    """)
    st.info("Use the sidebar to explore features.")

# =========================
# SINGLE PREDICTION
# =========================
elif menu == "🔍 Single Prediction":

    st.subheader("🔍 Predict Customer Churn")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure (months)", 0, 100)
        monthly = st.number_input("Monthly Charges", 0.0, 10000.0)
        total = st.number_input("Total Charges", 0.0, 100000.0)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    input_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "gender": gender,
        "InternetService": internet,
        "Contract": contract
    }

    if st.button("🔍 Predict"):

        try:
            processed = preprocess_input(input_data)

            pred = model.predict(processed)[0]
            prob = model.predict_proba(processed)[0][1]

            st.progress(int(prob * 100))

            if pred == 1:
                st.error(f"⚠️ High Risk of Churn ({prob:.2%})")
            else:
                st.success(f"✅ Low Risk (No Churn) ({prob:.2%})")

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# BATCH PREDICTION
# =========================
elif menu == "📂 Batch Prediction":

    st.subheader("📂 Upload CSV for Bulk Prediction")

    file = st.file_uploader("Upload your dataset", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write("Preview", df.head())

        try:
            processed = preprocess_input(df)

            df['Prediction'] = model.predict(processed)
            df['Probability'] = model.predict_proba(processed)[:,1]

            st.write("Results", df)

            st.download_button(
                "📥 Download Results",
                df.to_csv(index=False),
                "churn_results.csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# DATA VISUALIZATION
# =========================
elif menu == "📊 Data Visualization":

    st.subheader("📊 Data Insights")

    if os.path.exists("data/churn.csv"):
        df = pd.read_csv("data/churn.csv")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Churn Distribution")
            fig1, ax1 = plt.subplots()
            sns.countplot(x='Churn', data=df, ax=ax1)
            st.pyplot(fig1)

        with col2:
            st.write("Correlation Heatmap")

            # ✅ FIXED (NO ERROR)
            df_encoded = pd.get_dummies(df)

            fig2, ax2 = plt.subplots()
            sns.heatmap(df_encoded.corr(), cmap='coolwarm', ax=ax2)
            st.pyplot(fig2)

    else:
        st.warning("Dataset not found in data folder!")
