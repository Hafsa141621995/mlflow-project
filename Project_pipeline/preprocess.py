import argparse
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data, sep=";")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(columns=["y"])
    y = df["y"]

    os.makedirs(args.output, exist_ok=True)
    X.to_csv(os.path.join(args.output, "X.csv"), index=False)
    y.to_csv(os.path.join(args.output, "y.csv"), index=False)

if __name__ == "__main__":
    main()
