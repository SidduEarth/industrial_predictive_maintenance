"""Module: model_training.py

Trains and evaluates the Random Forest Classifier for predictive maintenance.
"""

import logging
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
  """Trains a Random Forest Classifier."""
  logging.info("Training Random Forest model...")
  model = RandomForestClassifier(
      n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
  )
  model.fit(X_train, y_train)
  logging.info("Model training complete.")
  return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
  """Evaluates model performance."""
  predictions = model.predict(X_test)
  probabilities = model.predict_proba(X_test)[:, 1]

  print("\n--- Classification Report ---")
  print(classification_report(y_test, predictions))

  auc_score = roc_auc_score(y_test, probabilities)
  logging.info(f"Test ROC-AUC Score: {auc_score:.4f}")
  return auc_score