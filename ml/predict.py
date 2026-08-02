import pandas as pd
import joblib
import os

# --- CONFIGURATION CONSTANTS ---
PREPROCESSOR_PATH = "saved_models/preprocessor.pkl"
MODEL_PATH = "saved_models/best_tuned_model.pkl"

# --- PIPELINE LOADING ---
def load_pipeline():
    """Loads and returns the preprocessor and trained model safely."""
    if not os.path.exists(PREPROCESSOR_PATH) or not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Artifacts missing. Please ensure train.py and tune.py have been run.")
    
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model

# --- PREDICTION ENGINE ---
def predict_churn(customer_data: dict, preprocessor, model) -> dict:
    """
    Takes a raw customer dictionary, preprocesses it, and returns the churn prediction.
    """
    #  Convert the raw dictionary into a Pandas DataFrame
    #  wrap the dictionary in a list [customer_data] because pandas requires it for single rows
    input_df = pd.DataFrame([customer_data])
    
    # Apply the exact same preprocessing used during training
    processed_features = preprocessor.transform(input_df)
    
    # Get the binary prediction (0 = Stay, 1 = Churn)
    prediction = int(model.predict(processed_features)[0])
    
    # Get the confidence probability (Risk Score)
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(processed_features)[0][1])
    else:
        # Fallback for algorithms that don't support probability
        probability = float(prediction)

    #  Return a clean, API-ready response dictionary
    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_level": "High Risk" if prediction == 1 else "Low Risk"
    }

# --- LOCAL TESTING (Only runs if you execute this file directly) ---
if __name__ == "__main__":
    
    # Sample new customer (Matches the features based on  SHAP plot insights)
    # Older, multiple products, inactive, from Germany = Highly likely to churn
    sample_customer = {
        "credit_score": 600,
        "country": "Germany",
        "gender": "Male",
        "age": 55,
        "tenure": 2,
        "balance": 125000.00,
        "products_number": 3,
        "credit_card": 1,
        "active_member": 0,       
        "estimated_salary": 85000.00
    }
    
    print("Loading machine learning pipeline...")
    prep, clf = load_pipeline()
    
    print("Scoring sample customer...")
    result = predict_churn(sample_customer, prep, clf)
    
    print("\n" + "="*35)
    print("        PREDICTION RESULT")
    print("="*35)
    print(f"Prediction:   {result['churn_prediction']}")
    print(f"Probability:  {result['churn_probability'] * 100:.2f}%")
    print(f"Risk Level:   {result['risk_level']}")
    print("="*35 + "\n")