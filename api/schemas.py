from pydantic import BaseModel, Field

class CustomerRequest(BaseModel):
    credit_score: int = Field(..., example=600, description="Customer's credit score")
    country: str = Field(..., example="Germany", description="Country of residence")
    gender: str = Field(..., example="Male", description="Male or Female")
    age: int = Field(..., example=55, description="Customer's age")
    tenure: int = Field(..., example=2, description="Years with the bank")
    balance: float = Field(..., example=125000.00, description="Account balance")
    products_number: int = Field(..., example=3, description="Number of bank products held")
    credit_card: int = Field(..., example=1, description="1 if holds credit card, 0 otherwise")
    active_member: int = Field(..., example=0, description="1 if active, 0 if inactive")
    estimated_salary: float = Field(..., example=85000.00, description="Estimated annual salary")

class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str