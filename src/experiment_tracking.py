
# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import json

from pathlib import Path
from datetime import datetime


# ============================================================
# 2. PROJECT PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORTS_DIR = PROJECT_ROOT / "reports"

METRICS_FILE = REPORTS_DIR / "final_model_comparison.csv"

PARAMETERS_FILE = REPORTS_DIR / "best_model_parameters.json"

OUTPUT_FILE = REPORTS_DIR / "experiment_tracking.csv"


# ============================================================
# 3. LOAD MODEL EVALUATION METRICS
# ============================================================

metrics_df = pd.read_csv(METRICS_FILE)

print("Model evaluation metrics loaded successfully!")


# ============================================================
# 4. LOAD BEST MODEL PARAMETERS
# ============================================================

with open(PARAMETERS_FILE, "r") as file:

    best_parameters = json.load(file)

print("Best model parameters loaded successfully!")


# ============================================================
# 5. GENERATE EXPERIMENT TRACKING RECORDS
# ============================================================

experiment_records = []

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for index, row in metrics_df.iterrows():

    model_name = row["Model"]

    # Use Testing Accuracy when Accuracy is unavailable
    accuracy = row["Accuracy"]

    if pd.isna(accuracy):
        accuracy = row["Testing Accuracy"]

    # Attach parameters to the optimized model
    if model_name == "Optimized Random Forest":
        model_parameters = best_parameters
    else:
        model_parameters = {}

    # Identify selected model
    selected_model = (
        "Yes"
        if model_name == "Optimized Random Forest"
        else "No"
    )

    experiment = {

        "Experiment_ID": f"EXP_{index + 1:03d}",

        "Timestamp": timestamp,

        "Model_Name": model_name,

        "Accuracy": accuracy,

        "Precision": row["Precision"],

        "Recall": row["Recall"],

        "F1_Score": row["F1 Score"],

        "ROC_AUC": row["ROC-AUC"],

        "Training_Accuracy": row["Training Accuracy"],

        "Testing_Accuracy": row["Testing Accuracy"],

        "Model_Parameters": json.dumps(model_parameters),

        "Selected_Model": selected_model

    }

    experiment_records.append(experiment)


# ============================================================
# 6. CREATE EXPERIMENT TRACKING DATAFRAME
# ============================================================

experiment_df = pd.DataFrame(experiment_records)


# ============================================================
# 7. SAVE EXPERIMENT TRACKING REPORT
# ============================================================

experiment_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\nExperiment Tracking Completed Successfully!")

print("\nExperiment Tracking Results:")

print(experiment_df.to_string(index=False))

print("\nReport saved at:")

print(OUTPUT_FILE)
