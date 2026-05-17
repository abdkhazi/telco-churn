import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean():
    # Load data
    df = pd.read_csv(r"C:\Users\abdul\PycharmProjects\PythonProject2\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")

    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Missing values:\n{df.isnull().sum()}")

    # TotalCharges has spaces instead of nulls — fix it
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    # Drop customerID — not useful for prediction
    df.drop(columns=["customerID"], inplace=True)

    # Encode target variable
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode binary categorical columns
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService",
                   "PaperlessBilling"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0,
                                "Male": 1, "Female": 0})

    # One-hot encode remaining categorical columns
    df = pd.get_dummies(df, drop_first=True)

    print(f"\nFinal shape: {df.shape}")
    print(f"Churn distribution:\n{df['Churn'].value_counts()}")

    return df

def split_data(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_and_clean()
    X_train, X_test, y_train, y_test = split_data(df)
    print("\nPreprocessing complete.")


