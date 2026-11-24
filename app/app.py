
import streamlit as st
import pandas as pd
import joblib
import warnings
import sys
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add src directory to Python path
sys.path.append(str(PROJECT_ROOT / "src"))

# Import prediction logging function
from logging_config import log_prediction

# Suppress known feature-name compatibility warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "deployment_pipeline.joblib"

# Load trained pipeline
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

deployment_pipeline = load_model()

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

# Application heading
st.title("🏦 Bank Marketing Campaign Prediction")

st.write(
    "Predict whether a customer is likely to subscribe "
    "to a bank term deposit."
)

st.divider()

# Customer input form
st.subheader("Enter Customer Details")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        job = st.selectbox(
            "Job",
            [
                "admin.", "blue-collar", "entrepreneur",
                "housemaid", "management", "retired",
                "self-employed", "services", "student",
                "technician", "unemployed", "unknown"
            ]
        )

        marital = st.selectbox(
            "Marital Status",
            ["married", "single", "divorced"]
        )

        education = st.selectbox(
            "Education",
            ["primary", "secondary", "tertiary", "unknown"]
        )

        default = st.selectbox(
            "Has Credit in Default?",
            ["no", "yes"]
        )

        balance = st.number_input(
            "Account Balance",
            value=1000
        )

        housing = st.selectbox(
            "Housing Loan?",
            ["no", "yes"]
        )

        loan = st.selectbox(
            "Personal Loan?",
            ["no", "yes"]
        )

    with col2:
        contact = st.selectbox(
            "Contact Communication Type",
            ["cellular", "telephone", "unknown"]
        )

        day = st.number_input(
            "Last Contact Day",
            min_value=1,
            max_value=31,
            value=15
        )

        month = st.selectbox(
            "Last Contact Month",
            [
                "jan", "feb", "mar", "apr", "may",
                "jun", "jul", "aug", "sep", "oct",
                "nov", "dec"
            ]
        )

        campaign = st.number_input(
            "Number of Contacts During Campaign",
            min_value=1,
            value=1
        )

        pdays = st.number_input(
            "Days Since Previous Contact (-1 if never contacted)",
            min_value=-1,
            value=-1
        )

        previous = st.number_input(
            "Previous Contacts",
            min_value=0,
            value=0
        )

        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            ["unknown", "other", "failure", "success"]
        )

        previously_contacted = st.selectbox(
            "Previously Contacted?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    submit_button = st.form_submit_button(
        "Predict Subscription"
    )

# Prediction
if submit_button:

    customer_data = pd.DataFrame(
        [[
            age, job, marital, education, default,
            balance, housing, loan, contact, day,
            month, campaign, pdays, previous,
            poutcome, previously_contacted
        ]],
        columns=[
            "age", "job", "marital", "education",
            "default", "balance", "housing", "loan",
            "contact", "day", "month", "campaign",
            "pdays", "previous", "poutcome",
            "previously_contacted"
        ]
    )

    prediction = deployment_pipeline.predict(customer_data)[0]

    probabilities = deployment_pipeline.predict_proba(
        customer_data
    )[0]

    # Record prediction in log file
    log_prediction(
        customer_data.iloc[0].to_dict(),
        prediction,
        probabilities
    )

    # Display results
    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Customer is likely to subscribe!")
    else:
        st.warning("Customer is unlikely to subscribe.")

    st.write("### Prediction Probabilities")

    probability_df = pd.DataFrame({
        "Result": ["No Subscription", "Subscription"],
        "Probability": probabilities
    })

    st.bar_chart(
        probability_df.set_index("Result")
    )

    st.write(
        f"**Subscription probability:** "
        f"{probabilities[1] * 100:.2f}%"
    )

    st.write(
        f"**No subscription probability:** "
        f"{probabilities[0] * 100:.2f}%"
    )
