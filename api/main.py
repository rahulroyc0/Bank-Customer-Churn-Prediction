from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
import os
import sys

# ------ PATH RESOLUTION -------
# Ensures Python can find the 'ml' folder from the root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


from ml.predict import predict_churn
from api.schemas import CustomerRequest, PredictionResponse

# --- GLOBAL VARIABLES FOR ML MODELS ---
preprocessor = None
model = None

# Absolute paths to your ML artifacts
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "ml", "saved_models", "preprocessor.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "ml", "saved_models", "best_tuned_model.pkl")


# --- LIFESPAN MANAGER  ---
# This ensures models are loaded into memory exactly ONCE when the server starts
# rather than loading them every single time a user clicks "Predict".
@asynccontextmanager
async def lifespan(app: FastAPI):
    global preprocessor, model
    
    print("Starting up: Loading Machine Learning artifacts...")
    if not os.path.exists(PREPROCESSOR_PATH) or not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model artifacts missing. Run train.py and tune.py first.")
    
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    print("Models loaded successfully. API is ready to accept requests.")

    yield
    

# --- APP INITIALIZATION ---
app = FastAPI(
    title="Bank Churn Prediction API",
    description="Backend API serving an XGBoost Model for Customer Churn Prediction.",
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS CONFIGURATION FOR NEXT.JS ---
# It allows your Next.js app (Port 3000) to talk to FastAPI (Port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, PUT, DELETE
    allow_headers=["*"],  # Allows all headers
)

# --- API ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "ok", "message": "API is running!"}

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(customer: CustomerRequest):
    # Receives JSON data from the Next.js frontend, passes it to the ML model, and returns a Churn Risk score.
    
    try:
        # Convert Pydantic object to a standard dictionary
        customer_data = customer.model_dump() 
        
        # Call the ML function
        result = predict_churn(customer_data, preprocessor, model)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    