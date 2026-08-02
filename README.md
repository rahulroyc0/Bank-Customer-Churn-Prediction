#  AI-Powered Bank Customer Churn Prediction System

An end-to-end Machine Learning web application that predicts whether a bank customer is likely to churn (leave the bank). The project compares multiple machine learning algorithms, automatically selects the best baseline model, performs hyperparameter tuning, explains model predictions using SHAP, and serves real-time predictions through a FastAPI backend with a modern Next.js dashboard.

---

#  Overview

Customer churn is one of the biggest challenges faced by banks. Losing an existing customer is significantly more expensive than acquiring a new one.

This project predicts customer churn using machine learning so that banks can proactively identify high-risk customers and take preventive actions.

---

#  Features

- End-to-end Machine Learning pipeline
- Data preprocessing using Scikit-Learn Pipeline & ColumnTransformer
- Multiple baseline models
  - Logistic Regression
  - Random Forest
  - XGBoost
- Automatic best model selection using F1 Score
- Hyperparameter tuning using GridSearchCV
- SHAP Explainability for feature importance
- FastAPI REST API
- Interactive Next.js Dashboard
- Type-safe API validation using Pydantic
- Real-time churn prediction
- Churn probability score
- High Risk / Low Risk classification

---

#  Machine Learning Workflow

```text
Raw Dataset
      │
      ▼
Exploratory Data Analysis (EDA)
      │
      ▼
Data Preprocessing
      │
      ├── Train/Test Split
      ├── StandardScaler
      ├── OneHotEncoder
      └── ColumnTransformer
      │
      ▼
Train Baseline Models
      │
      ├── Logistic Regression
      ├── Random Forest
      └── XGBoost
      │
      ▼
Model Evaluation
      │
      ├── Accuracy
      ├── Precision
      ├── Recall
      ├── F1 Score
      ├── Confusion Matrix
      └── Classification Report
      │
      ▼
Best Baseline Model Selection
      │
      ▼
GridSearchCV Hyperparameter Tuning
      │
      ▼
Best Tuned Model
      │
      ▼
SHAP Explainability
      │
      ▼
FastAPI Backend
      │
      ▼
Next.js Dashboard
```

---

#  Model Evaluation

Three machine learning algorithms were trained and compared:

- Logistic Regression
- Random Forest
- XGBoost

The best baseline model was selected based on the **F1 Score**, which provides a balanced evaluation of Precision and Recall for customer churn prediction.

Evaluation metrics used:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

#  Model Explainability

To improve model transparency, SHAP (SHapley Additive exPlanations) was used.

SHAP helps explain:

- Which features increase churn risk
- Which features decrease churn risk
- Overall feature importance
- Individual prediction explanations

This makes the model more interpretable for business users.

---

#  System Architecture

```text
                Next.js Dashboard
                        │
                        │ HTTP Request
                        ▼
                 FastAPI Backend
                        │
                        │
                        ▼
              Data Preprocessing
                        │
                        ▼
              Tuned XGBoost Model
                        │
                        ▼
              Churn Prediction
                        │
                        ▼
             Prediction Response
                        │
                        ▼
                Next.js Dashboard
```

---

#  Tech Stack

## Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS

---

## Backend

- FastAPI
- Python

---

## Machine Learning

- Scikit-Learn
- XGBoost
- Pandas
- NumPy
- SHAP
- Joblib

---

## Deployment

- Vercel (Frontend)
- Render (Backend)

---

#  Project Structure

```text
Bank-Customer-Churn-Prediction
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── ml/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tune.py
│   ├── explain.py
│   ├── predict.py
│   └── saved_models/
│
├── data/
│
├── frontend/
│   ├── src/
│   ├── app/
│   ├── package.json
│   └── ...
│
├── requirements.txt
│
└── README.md
```

---

#  API Endpoints

## Health Check

```
GET /
```

Response

```json
{
    "status": "ok",
    "message": "API is running!"
}
```

---

## Predict Customer Churn

```
POST /predict
```

Example Request

```json
{
    "credit_score": 600,
    "country": "Germany",
    "gender": "Male",
    "age": 55,
    "tenure": 2,
    "balance": 125000,
    "products_number": 3,
    "credit_card": 1,
    "active_member": 0,
    "estimated_salary": 85000
}
```

Example Response

```json
{
    "churn_prediction": 1,
    "churn_probability": 0.91,
    "risk_level": "High Risk"
}
```

---

#  Dataset

Dataset used:

**Bank Customer Churn Prediction Dataset**

Features include:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Member Status
- Estimated Salary

Target Variable

```
Churn
```

- 0 → Customer stays
- 1 → Customer leaves

---

#  Local Installation

## Clone Repository

```bash
git clone https://github.com/rahulroyc0/Bank-Customer-Churn-Prediction.git

cd Bank-Customer-Churn-Prediction
```

---

## Backend Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate

pip install -r requirements.txt

uvicorn api.main:app --reload
```

Backend

```
http://localhost:8000
```

Swagger Documentation

```
http://localhost:8000/docs
```

---

## Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

Frontend

```
http://localhost:3000
```

---

# 🌍 Deployment

## Frontend

Vercel

```
https://bank-customer-churn-prediction-gilt.vercel.app
```

---

## Backend

Render

```
https://bank-customer-churn-prediction-ow0a.onrender.com
```

Swagger

```
https://bank-customer-churn-prediction-ow0a.onrender.com/docs
```

---

# Screenshots

## Dashboard

<img width="1920" height="1080" alt="Screenshot (550)" src="https://github.com/user-attachments/assets/bd5d2a38-6044-47c5-9f6b-f2f1b9f81c4e" />


---

## Prediction Result

<img width="1920" height="1080" alt="Screenshot (551)" src="https://github.com/user-attachments/assets/22d97782-c80e-417b-b041-97089a8d0d32" />


---

## SHAP Feature Importance

<img width="2300" height="1739" alt="shap_summary_plot" src="https://github.com/user-attachments/assets/bc6a1984-50ad-4016-b225-3c2c923b2269" />


---

## FastAPI Swagger Documentation

<img width="1920" height="1080" alt="Screenshot (552)" src="https://github.com/user-attachments/assets/feffd1d7-e960-45b4-8112-23feb3d819be" />


---

#  Future Improvements

- Docker Support
- User Authentication
- Database Integration
- Prediction History
- Admin Dashboard
- CI/CD Pipeline
- Cloud Model Storage
- Model Monitoring
- Drift Detection
- Batch Predictions

---

#  Author

**Rahul Roy**

Production Engineering Undergraduate  
Jadavpur University

GitHub:

https://github.com/rahulroyc0


---

#  If you found this project useful, please consider giving it a Star.
