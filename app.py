import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Churn Prediction", layout="wide")

st.markdown("<h1 style='text-align: center;'>📊 Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)

# ======================
# LOAD MODEL
# ======================
if not os.path.exists("model/churn_model.pkl"):
    st.error("Run train_model.py first!")
    st.stop()

model = pickle.load(open("model/churn_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

# ======================
# SIDEBAR
# ======================
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction", "Data Visualization"])

# ======================
# FUNCTION
# ======================
def preprocess_input(input_df):
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)
    input_df = scaler.transform(input_df)
    return input_df

# ======================
# 1️⃣ SINGLE PREDICTION
# ======================
if option == "Single Prediction":

    st.subheader("🔍 Predict Single Customer")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure", 0, 100)
        monthly = st.number_input("Monthly Charges", 0.0, 10000.0)
        total = st.number_input("Total Charges", 0.0, 100000.0)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    input_df = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "gender": gender,
        "InternetService": internet,
        "Contract": contract
    }])

    if st.button("Predict"):
        data = preprocess_input(input_df)
        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        st.subheader("Result")

        st.progress(int(prob * 100))

        if pred == 1:
            st.error(f"⚠️ High Risk (Churn) - {prob:.2%}")
        else:
            st.success(f"✅ Low Risk (No Churn) - {prob:.2%}")

# ======================
# 2️⃣ BATCH PREDICTION
# ======================
elif option == "Batch Prediction":

    st.subheader("📂 Upload CSV for Bulk Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.write("Uploaded Data", df.head())

        processed = preprocess_input(df)

        df['Prediction'] = model.predict(processed)
        df['Probability'] = model.predict_proba(processed)[:,1]

        st.write("Prediction Result", df)

        st.download_button("Download Results", df.to_csv(index=False), "results.csv")

# ======================
# 3️⃣ VISUALIZATION
# ======================
elif option == "Data Visualization":

    st.subheader("📊 Data Visualization")

    if os.path.exists("data/churn.csv"):
        df = pd.read_csv("data/churn.csv")

        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots()
            sns.countplot(x='Churn', data=df, ax=ax1)
            st.pyplot(fig1)

        with col2:
            df['Churn'] = df['Churn'].map({'Yes':1,'No':0})
            fig2, ax2 = plt.subplots()
            sns.heatmap(df.corr(), cmap='coolwarm', ax=ax2)
            st.pyplot(fig2)

    else:
        st.warning("Dataset not found!")