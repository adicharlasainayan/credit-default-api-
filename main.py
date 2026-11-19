from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
package = joblib.load(BASE_DIR / "model_package.joblib")
model = package["model"]
THRESHOLD = float(package["threshold"])
FEATURES = package["feature_names"]

COST_FN = 500  # cost of a missed default
COST_FP = 50   # cost of rejecting a good applicant

app = FastAPI(
    title="Credit Default Risk Prediction (Cost-Sensitive)",
    description="Random Forest + cost-optimised decision threshold (UCI Credit Card Default dataset).",
    version="1.0.0",
)


class Applicant(BaseModel):
    LIMIT_BAL: float = Field(..., examples=[20000])
    SEX: int = Field(..., examples=[2])
    EDUCATION: int = Field(..., examples=[2])
    MARRIAGE: int = Field(..., examples=[1])
    AGE: int = Field(..., examples=[24])
    PAY_0: int = Field(..., examples=[2])
    PAY_2: int = Field(..., examples=[2])
    PAY_3: int = Field(..., examples=[-1])
    PAY_4: int = Field(..., examples=[-1])
    PAY_5: int = Field(..., examples=[-2])
    PAY_6: int = Field(..., examples=[-2])
    BILL_AMT1: float = Field(..., examples=[3913])
    BILL_AMT2: float = Field(..., examples=[3102])
    BILL_AMT3: float = Field(..., examples=[689])
    BILL_AMT4: float = Field(..., examples=[0])
    BILL_AMT5: float = Field(..., examples=[0])
    BILL_AMT6: float = Field(..., examples=[0])
    PAY_AMT1: float = Field(..., examples=[0])
    PAY_AMT2: float = Field(..., examples=[689])
    PAY_AMT3: float = Field(..., examples=[0])
    PAY_AMT4: float = Field(..., examples=[0])
    PAY_AMT5: float = Field(..., examples=[0])
    PAY_AMT6: float = Field(..., examples=[0])


@app.get("/")
def home():
    return {
        "message": "Credit Default Risk API is running",
        "docs": "/docs",
        "predict_endpoint": "POST /predict",
        "threshold": THRESHOLD,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/threshold")
def threshold_info():
    return {
        "optimal_threshold": THRESHOLD,
        "cost_missed_default": COST_FN,
        "cost_false_alarm": COST_FP,
        "features": FEATURES,
    }


@app.post("/predict")
def predict(applicant: Applicant):
    row = pd.DataFrame([applicant.model_dump()])[FEATURES]
    proba = float(model.predict_proba(row)[0, 1])
    default = int(proba >= THRESHOLD)
    return {
        "default_probability": round(proba, 4),
        "threshold": THRESHOLD,
        "prediction": default,
        "decision": "High risk - likely to default (reject / review)" if default
                    else "Low risk - approve",
    }
