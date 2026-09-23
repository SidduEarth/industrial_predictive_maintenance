"""Module: main.py

Orchestrates the entire machine learning pipeline execution.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing import clean_data
from src.model_training import evaluate_model, train_model


def main():
  print("=== Starting ML Pipeline Execution ===")

  # 1. Create dummy data for demonstration if file doesn't exist
  os.makedirs("data", exist_ok=True)
  os.makedirs("models", exist_ok=True)

  data_path = "data/sample_sensor_data.csv"
  if not os.path.exists(data_path):
    print("Creating sample synthetic sensor dataset...")
    import numpy as np

    np.random.seed(42)
    df = pd.DataFrame({
        "vibration_freq": np.random.normal(50, 10, 500),
        "temperature": np.random.normal(75, 15, 500),
        "pressure": np.random.normal(30, 5, 500),
        "machine_failure": np.random.choice([0, 1], size=500, p=[0.85, 0.15]),
    })
    df.to_csv(data_path, index=False)

  # 2. Load and Clean Data
  df = pd.read_csv(data_path)
  df_cleaned = clean_data(df)

  # 3. Split Features and Target
  X = df_cleaned.drop(columns=["machine_failure"])
  y = df_cleaned["machine_failure"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  # 4. Train and Evaluate Model
  model = train_model(X_train, y_train)
  evaluate_model(model, X_test, y_test)

  # 5. Save Artifact
  model_path = "models/random_forest_model.pkl"
  joblib.dump(model, model_path)
  print(f"Pipeline executed successfully. Model saved to {model_path}")


if __name__ == "__main__":
  main()