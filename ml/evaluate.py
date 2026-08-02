import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# File paths relative to ml/ folder
X_TEST_PATH = "../data/X_test.csv"
Y_TEST_PATH = "../data/y_test.csv"
PREPROCESSOR_PATH = "saved_models/preprocessor.pkl"
TARGET_BEST_BASELINE_PATH = "saved_models/best_baseline.pkl"

MODELS = {
    "Logistic Regression": "saved_models/logistic_regression.pkl",
    "Random Forest": "saved_models/random_forest.pkl",
    "XGBoost": "saved_models/xgboost.pkl"
}

def evaluate():
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)["churn"]

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_test_processed = preprocessor.transform(X_test)

    results = []
    loaded_models = {}

    for model_name, model_path in MODELS.items():
        # Load and store model reference in dictionary
        model = joblib.load(model_path)
        loaded_models[model_name] = model
        
        y_pred = model.predict(X_test_processed)

        # Collect key metrics
        results.append({
            "Model": model_name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1-Score": round(f1_score(y_test, y_pred), 4)
        })

    # Print clean comparison table
    comparison_df = pd.DataFrame(results)
    
    print("\n" + "="*55)
    print("           MODEL COMPARISON SUMMARY")
    print("="*55)
    print(comparison_df.to_string(index=False))
    print("="*55 + "\n")

    # --- Identify and save the best baseline model using JOBLIB ---
    best_idx = comparison_df["F1-Score"].idxmax()
    best_model_name = comparison_df.loc[best_idx, "Model"]
    best_f1 = comparison_df.loc[best_idx, "F1-Score"]
    
    print(f"CHAMPION MODEL: {best_model_name} (F1-Score: {best_f1})")
    
    # Save winning model object directly via joblib
    best_model_obj = loaded_models[best_model_name]
    joblib.dump(best_model_obj, TARGET_BEST_BASELINE_PATH)
    
    print(f"Saved champion baseline model directly to '{TARGET_BEST_BASELINE_PATH}' using joblib.")

if __name__ == "__main__":
    evaluate()