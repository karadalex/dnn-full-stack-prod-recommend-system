import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


MODEL_VERSION = 1

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

model.save(f"movie-recommender-small-v{MODEL_VERSION}")
