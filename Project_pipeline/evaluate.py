import argparse
import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.test_data, sep=";")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(columns=["y"])
    y = df["y"]

    model_path = os.path.join(args.model, "model.joblib")
    model = joblib.load(model_path)

    acc = model.score(X, y)
    print("Accuracy:", acc)

if __name__ == "__main__":
    main()
