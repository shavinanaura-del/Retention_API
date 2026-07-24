from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# load model
model = joblib.load("model_retention.pkl")

# load data 2026
data = pd.read_excel("2026 Clasify.xlsx")


@app.get("/")
def home():
    return {
        "message": "Retention API aktif"
    }


@app.get("/customers")
def customers():
    return data.to_dict(orient="records")


@app.get("/predict")
def predict():

    X = data[
        [
            "Segment_id",
            "Usia",
            "Closing",
            "RSP"
        ]
    ]

    prediction = model.predict(X)

    hasil = data.copy()
    hasil["Prediction"] = prediction

    return hasil.to_dict(orient="records")