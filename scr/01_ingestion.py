""" This script ingests the EV battery health prediction dataset, performs a train-test split, and saves the processed data to the specified folder. """

# import libaries 
from importlib.resources import path

import pandas as pd
import kagglehub
from pathlib import Path
from sklearn.model_selection import train_test_split

def ingest_data():
    # Download latest version
    path = kagglehub.dataset_download("srisyra02/ev-battery-health-prediction-dataset-20k")

    # Set the data folder path
    data_folder= Path(path)

    # Load the dataset into a pandas DataFrame
    df = pd.read_csv(data_folder / "ev battery_failure prediction Dataset.csv")

    # Train and test split
    X = df.drop(columns=["battery_failure", "vehicle_id"])
    y = df["battery_failure"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # output folder
    data = Path("data/processed")
    data.mkdir(parents=True, exist_ok=True)

    # Save Dataset 
    X_train.to_csv(data / "X_train.csv", index=False)
    X_test.to_csv(data / "X_test.csv", index=False)
    y_train.to_csv(data / "y_train.csv", index=False)
    y_test.to_csv(data / "y_test.csv", index=False)

    print("------------------------------")
    print("Processed data saved to:", data)
    print("------------------------------")

if __name__ == "__main__":
    ingest_data()

