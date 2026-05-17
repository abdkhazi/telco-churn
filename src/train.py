import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle

from preprocess import load_and_clean, split_data

def train_models(X_train, X_test, y_train, y_test):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False,
                                  eval_metric="logloss",
                                  random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Evaluate
        auc = roc_auc_score(y_test, y_prob)
        print(f"AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))

        results[name] = {"model": model, "auc": auc}

    return results

if __name__ == "__main__":
    df = load_and_clean()
    X_train, X_test, y_train, y_test = split_data(df)
    results = train_models(X_train, X_test, y_train, y_test)

    # Print summary
    print("\n=== MODEL COMPARISON ===")
    for name, res in results.items():
        print(f"{name}: AUC = {res['auc']:.4f}")