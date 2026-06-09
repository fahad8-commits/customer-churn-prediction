# Customer Churn Prediction Platform

## Overview

A machine learning-based Customer Churn Prediction Platform that predicts whether a telecom customer is likely to leave the service. The project includes data preprocessing, feature engineering, class imbalance handling using SMOTE, machine learning model comparison, FastAPI-based prediction API, ETL pipeline, and Tableau dashboard visualizations.

---

## Project Objectives

- Predict customer churn using machine learning models
- Handle class imbalance using SMOTE
- Compare multiple classification algorithms
- Deploy a REST API for real-time predictions
- Build business intelligence dashboards using Tableau
- Create an ETL pipeline for automated data processing

---

## Dataset

Dataset: Telco Customer Churn Dataset

Records: ~7,000 customers

Target Variable: Churn (Yes/No)

### Features

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges

---

## Project Architecture

Raw Data

↓

Data Cleaning

↓

Feature Engineering

↓

Train/Test Split

↓

SMOTE Balancing

↓

Model Training

↓

Model Evaluation

↓

FastAPI REST API

↓

Tableau Dashboard

---

## Technologies Used

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Machine Learning

- Scikit-Learn
- XGBoost
- Imbalanced-Learn (SMOTE)

### Visualization

- Matplotlib
- Tableau

### API Development

- FastAPI
- Uvicorn

### Version Control

- Git
- GitHub

---

## Machine Learning Models

### Logistic Regression

Baseline classification model used for churn prediction.

### Decision Tree

Tree-based classification model for interpretability.

### Random Forest

Ensemble model with hyperparameter tuning.

### XGBoost

Gradient boosting model for enhanced predictive performance.

---

## Model Files

text models/ ├── logistic_regression.pkl ├── decision_tree.pkl ├── random_forest.pkl ├── random_forest_baseline.pkl ├── random_forest_tuned.pkl ├── random_forest_engineered.pkl └── xgboost_model.pkl 

---

## Data Preprocessing

### Completed Steps

- Missing value handling
- Data cleaning
- Categorical encoding
- Feature engineering
- Train/Test split
- Feature scaling
- SMOTE oversampling

---

## ETL Pipeline

The ETL pipeline performs:

### Extract

- Load customer churn dataset

### Transform

- Data cleaning
- Missing value handling
- Feature preparation

### Load

- Store processed datasets for model training and dashboard analysis

---

## Tableau Dashboard

Dashboard Components:

### KPI Cards

- Total Customers
- Total Churned Customers
- Churn Rate
- Average Monthly Charges

### Business Analysis

- Churn Overview
- Contract Analysis
- Monthly Charges Analysis
- Tenure Analysis

---

## FastAPI REST API

### Run API

bash uvicorn api:app --reload 

### Example Endpoint

http POST /predict 

Returns churn prediction results for new customer data.

---

## Project Structure

text customer-churn-prediction/  ├── data/ ├── etl/ ├── models/ ├── notebooks/ ├── src/ ├── tests/ ├── api.py ├── requirements.txt ├── README.md 

---

## Key Achievements

- Built end-to-end customer churn prediction pipeline
- Implemented SMOTE for class imbalance handling
- Trained and compared multiple machine learning models
- Developed FastAPI REST API for real-time inference
- Created Tableau business intelligence dashboard
- Implemented ETL workflow for automated preprocessing
- Managed project using Git and GitHub

---

## Future Improvements

- Cloud deployment
- AWS S3 integration
- Docker containerization
- CI/CD pipeline
- Automated model retraining
- Streamlit web application

---

## Author

Fahad J

Data Science Student | Machine Learning Enthusiast | AI Developer