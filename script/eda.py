"""
eda.py
------
Day 2, Part 2: Exploratory Data Analysis on the cleaned churn dataset.

Goal: understand which customer traits are associated with churn,
before we build a prediction model in Day 3.

Outputs:
- summary_churn_by_contract.csv
- summary_churn_by_internet.csv
- summary_churn_by_payment.csv
- summary_stats.csv (overall numeric summary)
"""

import pandas as pd

df = pd.read_csv("/home/claude/project2/data/customer_churn_cleaned.csv")

overall_churn_rate = (df["Churn"] == "Yes").mean()
print(f"Overall churn rate: {overall_churn_rate:.1%}\n")

# ---------------------------------------------------------
# Churn rate by Contract type
# ---------------------------------------------------------
by_contract = df.groupby("Contract")["Churn"].apply(
    lambda x: (x == "Yes").mean()
).sort_values(ascending=False).reset_index()
by_contract.columns = ["Contract", "ChurnRate"]
print("Churn rate by Contract type:")
print(by_contract.to_string(index=False))
by_contract.to_csv("/home/claude/project2/data/summary_churn_by_contract.csv", index=False)

# ---------------------------------------------------------
# Churn rate by Internet Service
# ---------------------------------------------------------
by_internet = df.groupby("InternetService")["Churn"].apply(
    lambda x: (x == "Yes").mean()
).sort_values(ascending=False).reset_index()
by_internet.columns = ["InternetService", "ChurnRate"]
print("\nChurn rate by Internet Service:")
print(by_internet.to_string(index=False))
by_internet.to_csv("/home/claude/project2/data/summary_churn_by_internet.csv", index=False)

# ---------------------------------------------------------
# Churn rate by Payment Method
# ---------------------------------------------------------
by_payment = df.groupby("PaymentMethod")["Churn"].apply(
    lambda x: (x == "Yes").mean()
).sort_values(ascending=False).reset_index()
by_payment.columns = ["PaymentMethod", "ChurnRate"]
print("\nChurn rate by Payment Method:")
print(by_payment.to_string(index=False))
by_payment.to_csv("/home/claude/project2/data/summary_churn_by_payment.csv", index=False)

# ---------------------------------------------------------
# Tenure & charges: churned vs stayed
# ---------------------------------------------------------
numeric_summary = df.groupby("Churn")[["TenureMonths", "MonthlyCharges", "TotalCharges"]].mean().reset_index()
print("\nAverage Tenure/Charges by Churn status:")
print(numeric_summary.to_string(index=False))
numeric_summary.to_csv("/home/claude/project2/data/summary_stats.csv", index=False)
