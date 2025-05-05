import pandas as pd
from model import train_model, predict

def test_train_model():
    data_path = 'tests/data/train.csv'
    model = train_model(data_path)

    # Add assertions to check if the model was trained correctly


def test_predict():
    data_path = 'tests/data/train.csv'
    model = train_model(data_path)

    test_data = pd.read_csv(data_path).drop('target', axis=1)
    predictions = predict(model, test_data)

    # Add assertions to check if the predictions are correct
