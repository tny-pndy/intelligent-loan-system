import streamlit as st
import pandas as pd
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Smart Loan AI",
    page_icon="🏦",
    layout="wide"
)

# ================= LOAD MODEL =================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ================= CUSTOM CSS =================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #071120, #0f172a, #111827);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 60px;
    font-weight: 800;
    background: linear-gradient(to right, #06b6d4, #3b82f6, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 35px;
}

.glass-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(16px);
    padding: 28px;
    border-radius: 28px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.35);
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 18px;
    color: #f8fafc;
}

.stNumberInput label,
.stSelectbox label {
    color: #e2e8f0 !important;
    font-weight: 500 !important;
}

.stNumberInput input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: white !important;
    border-radius: 14px !important;
}

.stSelectbox > div > div {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

.stButton > button {
    width: 100%;
    height: 70px;
    border-radius: 20px;
    border: none;
    font-size: 24px;
    font-weight: 700;
    color: white;
    background: linear-gradient(90deg,#06b6d4,#3b82f6,#8b5cf6,#ec4899);
    transition: 0.4s ease;
    margin-top: 25px;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0px 0px 30px rgba(139,92,246,0.5);
}

.result-success {
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.5);
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    font-size: 34px;
    font-weight: 700;
    color: #22c55e;
    margin-top: 25px;
}

.result-fail {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.5);
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    font-size: 34px;
    font-weight: 700;
    color: #ef4444;
    margin-top: 25px;
}

.footer {
    text-align:center;
    color:#94a3b8;
    margin-top:40px;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="main-title">🏦 Smart Loan AI Predictor</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Modern Intelligent Loan Approval Prediction System</div>',
    unsafe_allow_html=True
)

# ================= FORM =================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown('<div class="section-title">💰 Financial Details</div>', unsafe_allow_html=True)

    applicant_income = st.number_input("Applicant Income", value=50000)
    coapplicant_income = st.number_input("Coapplicant Income", value=10000)
    savings = st.number_input("Savings", value=50000)
    collateral_value = st.number_input("Collateral Value", value=200000)
    loan_amount = st.number_input("Loan Amount", value=150000)
    dti_ratio = st.number_input("DTI Ratio", value=25.0)

with col2:

    st.markdown('<div class="section-title">📊 Applicant Profile</div>', unsafe_allow_html=True)

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1)
    credit_score = st.number_input("Credit Score", value=750)
    existing_loans = st.number_input("Existing Loans", value=0)
    loan_term = st.number_input("Loan Term", value=12)

    education_level = st.selectbox(
        "Education Level",
        ["Graduate", "Not Graduate"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col3:

    st.markdown('<div class="section-title">🏢 Employment & Loan Info</div>', unsafe_allow_html=True)

    employment_status = st.selectbox(
        "Employment Status",
        ["Salaried", "Self-employed", "Unemployed"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Single"]
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Car", "Education", "Home", "Personal"]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Rural", "Semiurban", "Urban"]
    )

    employer_category = st.selectbox(
        "Employer Category",
        ["Government", "MNC", "Private", "Unemployed"]
    )

st.markdown('</div>', unsafe_allow_html=True)

# ================= PREDICTION =================
if st.button("✨ Predict Loan Status"):

    data = {

        'Applicant_Income': applicant_income,
        'Coapplicant_Income': coapplicant_income,
        'Age': age,
        'Dependents': dependents,
        'Credit_Score': credit_score,
        'Existing_Loans': existing_loans,
        'DTI_Ratio': dti_ratio,
        'Savings': savings,
        'Collateral_Value': collateral_value,
        'Loan_Amount': loan_amount,
        'Loan_Term': loan_term,

        'Education_Level': 1 if education_level == "Graduate" else 0,

        'Employment_Status_Salaried': 1 if employment_status == "Salaried" else 0,
        'Employment_Status_Self-employed': 1 if employment_status == "Self-employed" else 0,
        'Employment_Status_Unemployed': 1 if employment_status == "Unemployed" else 0,

        'Marital_Status_Single': 1 if marital_status == "Single" else 0,

        'Loan_Purpose_Car': 1 if loan_purpose == "Car" else 0,
        'Loan_Purpose_Education': 1 if loan_purpose == "Education" else 0,
        'Loan_Purpose_Home': 1 if loan_purpose == "Home" else 0,
        'Loan_Purpose_Personal': 1 if loan_purpose == "Personal" else 0,

        'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if property_area == "Urban" else 0,

        'Gender_Male': 1 if gender == "Male" else 0,

        'Employer_Category_Government': 1 if employer_category == "Government" else 0,
        'Employer_Category_MNC': 1 if employer_category == "MNC" else 0,
        'Employer_Category_Private': 1 if employer_category == "Private" else 0,
        'Employer_Category_Unemployed': 1 if employer_category == "Unemployed" else 0,
    }

    # ================= CREATE DATAFRAME =================
    input_df = pd.DataFrame([data])

    # ================= SCALE COMPLETE DATAFRAME =================
    input_scaled = scaler.transform(input_df)

    # ================= PREDICTION =================
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:

        st.markdown(
            '<div class="result-success">✅ Loan Approved Successfully</div>',
            unsafe_allow_html=True
        )

        st.balloons()

    else:

        st.markdown(
            '<div class="result-fail">❌ Loan Approval Rejected</div>',
            unsafe_allow_html=True
        )

# ================= FOOTER =================
st.markdown(
    '<div class="footer">Built with ❤️ using Streamlit & Machine Learning</div>',
    unsafe_allow_html=True
)