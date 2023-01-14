import streamlit as st
import pandas as pd
import numpy as np


st.title('The Movie Database - DNN Recommendation System')

def load_data(nrows):
  df1 = pd.read_csv("../datasets/ml-latest-small/movies.csv")
  df2 = pd.read_csv("../datasets/ml-latest-small/links.csv")
  df3 = pd.read_csv("../datasets/movie-genres-posters/MovieGenre.csv", nrows=nrows)
  
  df = df1.merge(df2, left_on="movieId", right_on="movieId")
  df = df.merge(df3, left_on="imdbId", right_on="imdbId")

  return df

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 1,000 rows of data into the dataframe.
data = load_data(1000)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

st.subheader('Raw data')
st.write(data)

st.subheader('IMDB Scores')
hist_values = np.histogram(data["IMDB Score"], bins=10, range=(0,10))[0]
st.bar_chart(hist_values)