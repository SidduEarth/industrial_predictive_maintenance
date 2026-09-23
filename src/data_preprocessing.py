import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
  """Cleans dataset by dropping duplicates and imputing missing values."""
  logging.info("Starting data cleaning process...")

  # Drop duplicates
  initial_shape = df.shape
  df = df.drop_duplicates()
  logging.info(
      f"Dropped {initial_shape[0] - df.shape[0]} duplicate rows."
  )

  # Impute numerical missing values with median
  numeric_cols = df.select_dtypes(include=[np.number]).columns
  for col in numeric_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

  logging.info("Missing values imputed successfully.")
  return df


def cap_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
  """Caps outliers using the Interquartile Range (IQR) method."""
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df[column] = np.clip(df[column], lower_bound, upper_bound)
  return df