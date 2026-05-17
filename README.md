# Telco Customer Churn Prediction

Predicting customer churn for a telecommunications company using 
machine learning.
## Problem
Identifying customers likely to churn before they leave, enabling 
targeted retention campaigns. Missing a churner (false negative) is 
more costly than a false alarm — so recall on the churn class is 
prioritised alongside AUC.

## Dataset
Telco Customer Churn dataset from Kaggle (7,032 customers, 20 features).  
Download: https://www.kaggle.com/code/emineyetm/telco-customer-churn/  
Place the CSV in the `data/` folder before running.

## Models and Results

| Model               | AUC    |
|---------------------|--------|
| Logistic Regression | 0.8363 |
| XGBoost             | 0.8196 |
| Random Forest       | 0.8150 |

Logistic Regression outperformed tree-based models, suggesting the 
churn signal in this dataset is largely linear in nature.

## Key Insights (SHAP analysis)

| Driver                    | Finding |
|---------------------------|---------|
| Contract type             | Two-year contracts reduce churn most — moving month-to-month customers to longer contracts is the highest-leverage retention action |
| Tenure                    | Churn risk is highest in early months — onboarding experience is critical |
| Monthly charges           | Higher charges correlate with higher churn — pricing sensitivity is real |
| Fibre optic internet      | Fibre customers churn more — possible service quality or pricing issue |

## How to Run

1. Download the dataset from the Kaggle link above and place in `data/`
2. Install dependencies: `pip install -r requirements.txt`
3. Run preprocessing: `python src/preprocess.py`
4. Train models: `python src/train.py`
5. Generate SHAP analysis: `python src/evaluate.py`

## Tech Stack
Python · pandas · scikit-learn · XGBoost · SHAP · matplotlib