from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.joblib")

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(features: list):
    X = np.array(features).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}
