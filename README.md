# Industrial Predictive Maintenance ML Pipeline

## Overview
An end-to-end Machine Learning pipeline designed to forecast industrial machinery failures using multi-sensor time-series telemetry data. This project covers data cleaning, outlier mitigation, feature engineering, model training, and evaluation.

## Project Structure
- `src/data_preprocessing.py`: Handles missing values, duplicates, and outlier capping (IQR).
- `src/model_training.py`: Trains and evaluates the Random Forest Classifier.
- `main.py`: Orchestrates the end-to-end execution pipeline.

## Installation & Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Run the pipeline:
    ```bash
    python main.py

