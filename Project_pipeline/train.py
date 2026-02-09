import argparse
import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    X = pd.read_csv(os.path.join(args.data, "X.csv"))
    y = pd.read_csv(os.path.join(args.data, "y.csv")).values.ravel()

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    os.makedirs(args.model, exist_ok=True)
    joblib.dump(model, os.path.join(args.model, "model.joblib"))

if __name__ == "__main__":
    main()
