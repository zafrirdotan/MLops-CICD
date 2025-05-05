import argparse
import pickle

import pandas as pd

from model import train_model, predict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--type', type=str, required=True, help='Train or Predict')
    return parser.parse_args()

def main():
    args = parse_args()

    if args.type == "Train":
        model = train_model(args.data_path)

        # Save model
        with open(args.model_path, 'wb') as f:
            pickle.dump(model, f)

    if args.type == "Predict":
        with open(args.model_path, 'rb') as f:
            model = pickle.load(f)
            data = pd.read_csv(args.data_path)
            prediction = predict(model,data)
            df = pd.DataFrame(prediction)
            df.to_csv("output/prediction.csv")


if __name__ == '__main__':
    main()