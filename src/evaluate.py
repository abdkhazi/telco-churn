import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

from preprocess import load_and_clean, split_data

def plot_shap(model, X_test):
    print("Generating SHAP plot...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("SHAP plot saved as shap_summary.png")

def print_business_insights(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Mean absolute SHAP value per feature
    feature_importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance": abs(shap_values).mean(axis=0)
    }).sort_values("importance", ascending=False)

    print("\n=== TOP 5 CHURN DRIVERS ===")
    for i, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

if __name__ == "__main__":
    df = load_and_clean()
    X_train, X_test, y_train, y_test = split_data(df)

    # Train XGBoost — best tree model for SHAP
    model = XGBClassifier(use_label_encoder=False,
                           eval_metric="logloss",
                           random_state=42)
    model.fit(X_train, y_train)

    plot_shap(model, X_test)
    print_business_insights(model, X_test)