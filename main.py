from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# load model
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


@app.post("/predict")
def predict(customer: Customer):

    # data dari Google Sheet masuk ke sini
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