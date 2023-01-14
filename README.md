DNN Full Stack Production(ish) Recommendation System
====================================================

## Description

- Deep Neural Network for a simple movie recommendation system, using MLOps and FullStack web practices
- Technologies & Libraries: tensorflow, keras, fastapi, huggingface, mlflow, lakefs, minio, nextjs, streamlit, click, pandas, scikitlearn, jupyter

## Requirements

- yarn/npm
- python3.8
- pipenv
- docker, docker-compose

## Instructions

The first time
```bash
pipenv shell
pipenv install
```

- To run docker services `docker-compose up -d`
- To run data gui `cd explore && streamlit run data_gui.py`

## TODOs

- [x] nextjs, show simple grid of movies
- [x] movie dataset from kaggle [https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- [ ] mlflow to log experiments: (hyperparameter tuning)
- [ ] tensorflow for dnn
- [ ] scikitlearn for metrics (accuracy, etc)
- [ ] celery to periodically run dnn experiments
- [x] fastapi for model serving
- [x] lakefs for dataset and model versioning
- [x] minio for object storage
- [ ] version datasets, models and API deployment of models
- [ ] StreamLit app [https://docs.streamlit.io/library/get-started/create-an-app](https://docs.streamlit.io/library/get-started/create-an-app)
- [x] jupyter notebooks for exploration
- [ ] click cli to clean/prepare data, make a huggingface dataset and other project scripts [click](https://click.palletsprojects.com/en/8.1.x/)


## Dataset links

- [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)
- [https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- [https://www.kaggle.com/datasets/neha1703/movie-genre-from-its-poster](https://www.kaggle.com/datasets/neha1703/movie-genre-from-its-poster)


## Other resources

- tutorial to follow [https://towardsdatascience.com/modern-recommendation-systems-with-neural-networks-3cc06a6ded2c](https://towardsdatascience.com/modern-recommendation-systems-with-neural-networks-3cc06a6ded2c)