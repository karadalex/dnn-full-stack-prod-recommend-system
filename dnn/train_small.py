import mlflow
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import os
from mlflow import MlflowClient


MODEL_VERSION_MAJOR = os.getenv("MODEL_VERSION_MAJOR", 0)
MODEL_VERSION_MINOR = os.getenv("MODEL_VERSION_MINOR", 1)
MODEL_VERSION_PATCH = os.getenv("MODEL_VERSION_PATCH", 0)
MODEL_VERSION = f"{MODEL_VERSION_MAJOR}.{MODEL_VERSION_MINOR}.{MODEL_VERSION_PATCH}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = f"movie-recommender-small-v{MODEL_VERSION}"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Register model name in the model registry
client = MlflowClient()
client.create_registered_model(MODEL_NAME)

mlflow.tensorflow.autolog()

X_train_df = pd.read_csv("X_train_small.csv")
X_test_df = pd.read_csv("X_test_small.csv")
Y_train_df = pd.read_csv("Y_train_small.csv")
Y_test_df = pd.read_csv("Y_test_small.csv")

X_train = X_train_df.to_numpy()
X_test = X_test_df.to_numpy()
Y_train = Y_train_df.to_numpy() / 10
Y_test = Y_test_df.to_numpy()

# design the neural network model
model = Sequential()
model.add(Dense(44, input_dim=X_train.shape[1], activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1))

# define the loss function and optimization algorithm
model.compile(
  loss='mse', 
  optimizer='adam'
)

# Display the model's architecture
model.summary()

# Train the model
history = model.fit(
  X_train,
  Y_train,
  batch_size=100,
  epochs=20,
  verbose=1
  # We pass some validation for
  # monitoring validation loss and metrics
  # at the end of each epoch
  # validation_data=(X_val, Y_val),
)

print(history.history)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(X_test, Y_test, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(X_test[:3]) * 10
print("predicted ratings:", predictions)
print("actual ratings:", Y_test[:3])

model.save(MODEL_NAME)
