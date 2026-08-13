# Hospital Readmission Risk Prediction

## Overview
This project is a machine learning pipeline designed to predict the risk of patient readmissions. It features comprehensive data engineering, class imbalance handling, and an interactive web interface for easy predictions.

## Key Technologies & Tools
* **Machine Learning Model:** XGBoost Classifier
* **Class Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique)
* **Data Processing:** Data standardization and feature engineering pipelines
* **Web Application:** Streamlit

## Project Details
In medical datasets, readmission cases are often the minority class. To build a robust and unbiased predictive model, **SMOTE** was utilized to balance the classes. The core predictions are powered by an **XGBoost** classifier, chosen for its high performance and accuracy with tabular data.

The final model is deployed using a **Streamlit** web interface, allowing users to input patient metrics and instantly receive a readmission risk prediction.

## Note on Datasets
Due to GitHub's file size limitations, the large processed CSV files (`X_train.csv`, `scaled_readmission_dataset.csv`) are excluded from this repository via `.gitignore`. 
