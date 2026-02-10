from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ML API – Notebook 4")

class Item(BaseModel):
    x: float
    y: float

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/predict")
def predict(item: Item):
    return {"result": item.x + item.y}
