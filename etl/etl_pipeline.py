import pandas as pd

print("Starting ETL Pipeline...")

# Extract
df = pd.read_csv("../data/raw/Telco-Customer-Churn.csv")

print("Dataset Loaded")
print(df.shape)

# Transform
df = df.dropna()

print("Missing values removed")

# Load
df.to_csv(
    "../data/processed/churn_processed.csv",
    index=False
)

print("Processed dataset saved")
print("ETL Pipeline Completed")# ETL module for customer churn project
