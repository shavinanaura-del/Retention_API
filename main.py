from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib

app = FastAPI()

# Load model
model = joblib.load("model_retention.pkl")


class Customer(BaseModel):
    Segment_id: int
    Usia: int
    Closing: int
    RSP: float


@app.get("/")
def home():
    return {
        "message": "Retention API aktif"
    }


# ==========================
# Prediksi 1 Customer
# ==========================
@app.post("/predict")
def predict(customer: Customer):

    X = pd.DataFrame([{
        "Segment_id": customer.Segment_id,
        "Usia": customer.Usia,
        "Closing": customer.Closing,
        "RSP": customer.RSP
    }])

    prediction = model.predict(X)[0]

    return {
        "Prediction": int(prediction),
        "Retention": "Balik" if prediction == 1 else "Tidak Balik"
    }


# ==========================
# Prediksi Banyak Customer Sekaligus
# ==========================
@app.post("/predict_batch")
def predict_batch(customers: List[Customer]):

    X = pd.DataFrame([
        {
            "Segment_id": c.Segment_id,
            "Usia": c.Usia,
            "Closing": c.Closing,
            "RSP": c.RSP
        }
        for c in customers
    ])

    predictions = model.predict(X)

    hasil = []

    for p in predictions:
        hasil.append({
            "Prediction": int(p),
            "Retention": "Balik" if p == 1 else "Tidak Balik"
        })

    return hasil