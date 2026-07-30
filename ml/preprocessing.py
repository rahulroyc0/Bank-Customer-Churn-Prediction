import pandas as pd
from sklearn.model_selection import train_test_split

# LOAD THE DATA SET
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# PREPROCESS THE DATA SET
def preprocess_data(df):

    # Remove customer id
    df = df.drop(columns=["customer_id"])

    # Features
    X = df.drop(columns=["churn"])

    # Target
    y = df["churn"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test