from typing import Union
from fastapi import FastAPI

app = FastAPI()


@app.get("/predict")
def model_prediction():
  return {"model_output": ""}
