from typing import Union
from fastapi import FastAPI
import tensorflow as tf
import pandas as pd
import numpy as np
from pydantic import BaseModel
import logging


movies_genres_df = pd.read_csv("../datasets/movies-genres-features.csv")

# TODO: Fetch trained model from a model registry or lakefs
latest_model = tf.keras.models.load_model("../dnn/movie-recommender-small-v1")

app = FastAPI()


@app.get("/model-config")
def model_config():
  return {"model_config": latest_model.get_config()}


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
