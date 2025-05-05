# model.py
import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model(data_path):
    # Load data
    data = pd.read_csv(data_path)
    X = data.drop('target', axis=1)
    y = data['target']

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    return model

def predict(model, data):
    # Make predictions
    predictions = model.predict(data)
    return predictions
