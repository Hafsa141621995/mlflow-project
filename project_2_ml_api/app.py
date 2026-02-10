from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import numpy as np

app = FastAPI()

model = joblib.load("model.joblib")

class InputData(BaseModel):
    x: float

@app.get("/")
def root():
    return {"message": "ML API is running"}

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict(np.array([[data.x]]))
    return {"prediction": float(prediction[0])}
