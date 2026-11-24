# Bank Marketing Campaign Prediction

## 1. Project Overview

The **Bank Marketing Campaign Prediction** project uses Machine Learning to predict whether a customer will subscribe to a bank's term deposit after a marketing campaign.

The objective is to help banks identify customers who are more likely to subscribe and improve the effectiveness of their marketing campaigns.

## 2. Problem Statement

Bank marketing campaigns involve contacting many customers, but not every customer subscribes to a term deposit.

This project builds a classification model to predict customer subscription outcomes using customer demographic information, financial details, and previous campaign interactions.

**Target Variable:** `y`

- `yes` – Customer subscribed.
- `no` – Customer did not subscribe.

## 3. Dataset Information

- Dataset: Bank Marketing Dataset
- Total Records: 45,211
- Original Features: 16 input features and 1 target variable
- Problem Type: Binary Classification

The `duration` feature was excluded because it is known only after the call and would cause data leakage during pre-call prediction.

## 4. Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib
- Streamlit
- Git and GitHub

## 5. Project Workflow

1. Data loading and understanding
2. Exploratory Data Analysis (EDA)
3. Data cleaning and preprocessing
4. Handling missing values
5. Categorical encoding using OneHotEncoder
6. Numerical scaling using StandardScaler
7. Train-test splitting
8. Baseline model development
9. Handling class imbalance using SMOTE
10. Hyperparameter optimization
11. Model evaluation and comparison
12. Building an end-to-end ML pipeline
13. Streamlit frontend development
14. Prediction logging and experiment tracking
15. Deployment preparation

## 6. Machine Learning Algorithms

The following algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gaussian Naive Bayes
- Optimized Random Forest

SMOTE was used to address class imbalance in the training data.

## 7. Final Model Results

The Optimized Random Forest was selected based on its F1-score.

| Metric | Optimized Random Forest |
|---|---:|
| Accuracy | 88.36% |
| Precision | 50.34% |
| Recall | 35.07% |
| F1 Score | 41.34% |
| ROC-AUC | 79.26% |

The optimized model achieved a better F1-score than the baseline Random Forest, improving the balance between precision and recall.

## 8. Model Optimization

RandomizedSearchCV was used to tune the Random Forest hyperparameters.

Best parameters:

- n_estimators: 100
- min_samples_split: 5
- min_samples_leaf: 2
- max_features: log2
- max_depth: None

## 9. Project Structure

```text
BANK_MARKET_CAMPAIGN_PREDICTION/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── logs/
│
├── models/
│   ├── best_bank_campaign_model.pkl
│   ├── deployment_pipeline.joblib
│   ├── preprocessor_sprint1.joblib
│   └── sprint2_baseline_models.pkl
│
├── notebooks/
│   ├── Sprint3_Optimization_and_Final_Model.ipynb
│   └── Sprint4_Deployment_and_MLOps.ipynb
│
├── reports/
│   ├── final_model_comparison.csv
│   ├── optimized_model_metrics.csv
│   ├── best_model_parameters.json
│   ├── experiment_tracking.csv
│   └── sprint3_summary.txt
│
├── src/
│   ├── preprocessing.py
│   ├── logging_config.py
│   └── experiment_tracking.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 10. Run the Project Locally

### Step 1: Clone the repository

```bash
git clone https://github.com/Yuvidev9/BANK_MARKET_CAMPAIGN_PREDICTION.git
```

### Step 2: Navigate to the project folder

```bash
cd BANK_MARKET_CAMPAIGN_PREDICTION
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the virtual environment

Windows CMD:

```cmd
.venv\Scripts\activate
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Streamlit application

```bash
streamlit run app/app.py
```

The application will open in your browser.

## 11. Deployment and MLOps

The project includes:

- End-to-end preprocessing and prediction pipeline.
- Streamlit-based user interface.
- Model loading using Joblib.
- GitHub version control.
- Experiment tracking using CSV reports.
- Prediction logging using Python logging.
- Model distribution through GitHub Releases.

## 12. Future Improvements

- Improve recall for customers who subscribe.
- Explore advanced ensemble algorithms such as XGBoost and LightGBM.
- Implement continuous model monitoring.
- Integrate MLflow for advanced experiment tracking.
- Improve cloud deployment and monitoring.

## 13. Author

**Yuvaraj Naredla**

GitHub: https://github.com/Yuvidev9

---

**Project:** Bank Marketing Campaign Prediction  
**Domain:** Machine Learning | Data Science | MLOps  
**Problem Type:** Binary Classification
