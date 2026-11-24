
# preprocessing.py

import joblib
import pandas as pd
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the existing Sprint 1 preprocessor
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor_sprint1.joblib"

preprocessor = joblib.load(PREPROCESSOR_PATH)

# Original customer input columns
NUMERICAL_COLUMNS = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
    "previously_contacted"
]

CATEGORICAL_COLUMNS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome"
]

INPUT_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS


def preprocess_input(customer_data):
    """
    Transform raw customer data using the saved Sprint 1 preprocessor.
    """

    # Convert input into DataFrame
    customer_df = pd.DataFrame(
        customer_data,
        columns=INPUT_COLUMNS
    )

    # Reorder columns to match the original preprocessor
    customer_df = customer_df[
        preprocessor.feature_names_in_
    ]

    # Apply saved preprocessing transformations
    processed_data = preprocessor.transform(customer_df)

    return processed_data
