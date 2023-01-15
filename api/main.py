from typing import Union
from fastapi import FastAPI
import tensorflow as tf
import pandas as pd
import numpy as np
from pydantic import BaseModel
import mlflow
from mlflow import MlflowClient
import os


MODEL_VERSION_MAJOR = os.getenv("MODEL_VERSION_MAJOR", 0)
MODEL_VERSION_MINOR = os.getenv("MODEL_VERSION_MINOR", 1)
MODEL_VERSION_PATCH = os.getenv("MODEL_VERSION_PATCH", 0)
MODEL_VERSION = f"{MODEL_VERSION_MAJOR}.{MODEL_VERSION_MINOR}.{MODEL_VERSION_PATCH}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = f"movie-recommender-small-v{MODEL_VERSION}"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

movies_genres_df = pd.read_csv("../datasets/movies-genres-features.csv")

app = FastAPI()


@app.get("/model-config")
def model_config():
  latest_model = mlflow.tensorflow.load_model(
    model_uri=f"models:/{MODEL_NAME}/latest"
  )
  return {"model_config": latest_model.get_config()}


@app.get("/latest-model-version")
def model_versions():
  return {"version": MODEL_VERSION}


class Input(BaseModel):
  user_id: int
  movie_id: int

@app.post("/predict")
def model_prediction(input: Input):
  latest_model = mlflow.tensorflow.load_model(
    model_uri=f"models:/{MODEL_NAME}/latest"
  )
  model_input = movies_genres_df[movies_genres_df.movieId == input.movie_id]
  model_input["userId"] = input.user_id
  model_input = model_input.astype("float64")
  print(model_input)

  prediction = round(10 * latest_model.predict(model_input)[0][0], 2)
  print(prediction)

  return {
    "model_output": prediction
  }
