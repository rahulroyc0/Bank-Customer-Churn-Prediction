import joblib
from preprocessing import load_data, preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# File paths relative to the ml/ folder
FILE_PATH = "../data/Bank Customer Churn Prediction.csv"

def train():
    # Load data
    print("Loading data...")
    df = load_data(FILE_PATH)

    # Get splits and preprocessor from your preprocessing script
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = preprocess_data(df)

    # Save test set to CSV in the data/ folder for evaluate.py
    X_test.to_csv("../data/X_test.csv", index=False)
    y_test.to_csv("../data/y_test.csv", index=False)
    print("Saved X_test.csv and y_test.csv to ../data/")

    # Preprocess training features
    print("Preprocessing training features...")
    X_train_processed = preprocessor.fit_transform(X_train)

    # Train Model 1: Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_processed, y_train)

    # Train Model 2: Random Forest Classifier
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_processed, y_train)

    # --- MODEL 3: XGBoost ---
    print("Training XGBoost Classifier...")
    xgb_model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42, 
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_processed, y_train)

    # Save all models and preprocessor
    print("\nSaving artifacts...")
    joblib.dump(preprocessor, "saved_models/preprocessor.pkl")
    joblib.dump(lr_model, "saved_models/logistic_regression.pkl")
    joblib.dump(rf_model, "saved_models/random_forest.pkl")
    joblib.dump(xgb_model, "saved_models/xgboost.pkl")

    print("All 3 models trained and saved successfully!")

if __name__ == "__main__":
    train()