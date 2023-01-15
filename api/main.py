from typing import Union
from fastapi import FastAPI
import tensorflow as tf
import pandas as pd
import numpy as np
from pydantic import BaseModel
import logging
import os


MODEL_VERSION_MAJOR = os.getenv("MODEL_VERSION_MAJOR", 0)
MODEL_VERSION_MINOR = os.getenv("MODEL_VERSION_MINOR", 1)
MODEL_VERSION_PATCH = os.getenv("MODEL_VERSION_PATCH", 0)
MODEL_VERSION = f"{MODEL_VERSION_MAJOR}.{MODEL_VERSION_MINOR}.{MODEL_VERSION_PATCH}"

movies_genres_df = pd.read_csv("../datasets/movies-genres-features.csv")

# TODO: Fetch trained model from a model registry or lakefs
latest_model = tf.keras.models.load_model(f"../dnn/movie-recommender-small-v{MODEL_VERSION}")

app = FastAPI()


@app.get("/model-config")
def model_config():
  return {"model_config": latest_model.get_config()}


@app.get("/latest-model-version")
def model_versions():
  return {"version": MODEL_VERSION}


class Input(BaseModel):
  user_id: int
  movie_id: int

@app.post("/predict")
def model_prediction(input: Input):
  model_input = np.append(movies_genres_df[movies_genres_df.movieId == input.movie_id].to_numpy(), [input.user_id])
  model_input = model_input.reshape((1,21))
  print(model_input)

  prediction = round(10 * latest_model.predict(model_input)[0][0], 2)
  print(prediction)

  return {
    "model_output": prediction
  }
