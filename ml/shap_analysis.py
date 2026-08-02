import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- CONFIGURATION CONSTANTS ---
X_TEST_PATH = "../data/X_test.csv"
PREPROCESSOR_PATH = "saved_models/preprocessor.pkl"
MODEL_PATH = "saved_models/best_tuned_model.pkl"
PLOT_OUTPUT_PATH = "shap_summary_plot.png"

# --- ARTIFACT LOADING ---
def load_artifacts(preprocessor_path, model_path):
    """Loads and returns the preprocessor and trained model."""
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    return preprocessor, model

# --- DATA PREPARATION ---
def prepare_data(X_test_path, preprocessor):
    """Loads test data, applies preprocessing, and formats column names."""
    X_test = pd.read_csv(X_test_path)
    X_processed = preprocessor.transform(X_test)
    
    # Safety check: Convert sparse matrix to dense array if needed
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
        
    # Clean the feature names by removing Scikit-Learn prefixes
    raw_names = preprocessor.get_feature_names_out()
    clean_names = [col.split("__")[-1] for col in raw_names]
    
    return pd.DataFrame(X_processed, columns=clean_names)

# --- SHAP CALCULATION (DYNAMIC ROUTING) ---
def calculate_shap_values(model, X_df):
    """Dynamically chooses the correct SHAP explainer based on model type."""
    # Identify the algorithm class name (e.g., 'XGBClassifier', 'LogisticRegression')
    model_type = type(model).__name__
    print(f"   -> Detected Model Type: {model_type}")
    
    # Route to the appropriate mathematical explainer
    if model_type in ["XGBClassifier", "RandomForestClassifier"]:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(X_df)
        
    elif model_type == "LogisticRegression":
        # Linear models require the background dataset (X_df) to calculate feature means
        explainer = shap.LinearExplainer(model, X_df)
        shap_values = explainer.shap_values(X_df)
        
    else:
        # Fallback for any other algorithm (SVM, KNN, Neural Networks, etc.)
        explainer = shap.Explainer(model, X_df)
        shap_values = explainer.shap_values(X_df)

    # --- Standardize the output format for plotting ---
    # If output is a list (Random Forest), grab index 1 (Churn class)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    # If output is a 3D array (Samples, Features, Classes), slice the Churn class
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]
        
    return shap_values

# --- VISUALIZATION ---
def save_shap_plot(shap_values, X_df, output_path):
    """Plots the SHAP summary and saves it to a file."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_df, show=False)
    
    plt.title("Feature Importance - Churn Drivers", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

# --- MAIN ORCHESTRATOR ---
def run_shap_pipeline():
    """Executes the loosely coupled functions in order."""
    print("Loading artifacts...")
    preprocessor, model = load_artifacts(PREPROCESSOR_PATH, MODEL_PATH)
    
    print("Preparing test data...")
    X_df = prepare_data(X_TEST_PATH, preprocessor)
    
    print("Calculating SHAP values dynamically...")
    shap_values = calculate_shap_values(model, X_df)
    
    print("Generating and saving plot...")
    save_shap_plot(shap_values, X_df, PLOT_OUTPUT_PATH)
    
    print(f"Success! SHAP plot saved to '{PLOT_OUTPUT_PATH}'")

if __name__ == "__main__":
    run_shap_pipeline()