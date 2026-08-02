import joblib
from sklearn.model_selection import GridSearchCV
from preprocessing import load_data, preprocess_data

DATA_PATH = "../data/Bank Customer Churn Prediction.csv"
BEST_BASELINE_PATH = "saved_models/best_baseline.pkl"
FINAL_TUNED_PATH = "saved_models/best_tuned_model.pkl"

#  Dictionary mapping for hyperparameter grids 
PARAM_GRIDS = {
    "LogisticRegression": {
        'C': [0.01, 0.1, 1, 10],
        'max_iter': [1000, 2000]
    },
    "RandomForestClassifier": {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    },
    "XGBClassifier": {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
}

def tune_best_model():
    print("Loading the champion baseline model...")
    base_model = joblib.load(BEST_BASELINE_PATH)
    
    # Identify algorithm class name
    model_type = base_model.__class__.__name__
    print(f"Detected Model Type: {model_type}")

    # Clean Dictionary Lookup 
    param_grid = PARAM_GRIDS.get(model_type)
    if not param_grid:
        raise ValueError(f"No hyperparameter grid configured for model type: '{model_type}'")

    print("\nLoading and preprocessing training data...")
    df = load_data(DATA_PATH)
    X_train, _, y_train, _, preprocessor = preprocess_data(df)
    X_train_processed = preprocessor.fit_transform(X_train)

    print("\nStarting Hyperparameter Tuning (GridSearchCV)...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_processed, y_train)

    print("\n" + "="*45)
    print("         TUNING COMPLETE")
    print("="*45)
    print(f"Best Tuned F1-Score: {grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")

    # Save final tuned model
    best_tuned_model = grid_search.best_estimator_
    joblib.dump(best_tuned_model, FINAL_TUNED_PATH)
    print(f"\n Final Tuned Model saved to '{FINAL_TUNED_PATH}'")

if __name__ == "__main__":
    tune_best_model()