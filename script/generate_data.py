"""
generate_data.py
-----------------
Generates a synthetic Customer Churn dataset (~5,000 rows) for a
subscription-based business, styled after the well-known Telco
Customer Churn dataset.

Includes realistic churn patterns AND intentional data quality
issues (missing values, inconsistent text casing, duplicate rows,
wrong data types) so we get practice cleaning it in Day 2.
"""

import numpy as np
import pandas as pd
import random

random.seed(42)
np.random.seed(42)

N = 5000

# ---------------------------------------------------------
# 1. Base customer attributes
# ---------------------------------------------------------
customer_id = [f"CUST-{10000+i}" for i in range(N)]
gender = np.random.choice(["Male", "Female"], size=N)
senior_citizen = np.random.choice([0, 1], size=N, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], size=N, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], size=N, p=[0.3, 0.7])

tenure_months = np.random.randint(0, 73, size=N)  # 0 to 72 months

phone_service = np.random.choice(["Yes", "No"], size=N, p=[0.9, 0.1])
internet_service = np.random.choice(
    ["DSL", "Fiber optic", "No"], size=N, p=[0.35, 0.45, 0.20]
)

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], size=N, p=[0.55, 0.24, 0.21]
)

payment_method = np.random.choice(
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    size=N,
)

paperless_billing = np.random.choice(["Yes", "No"], size=N, p=[0.6, 0.4])

# Monthly charges depend loosely on internet service type
base_charge = np.where(
    internet_service == "Fiber optic",
    np.random.normal(85, 15, N),
    np.where(internet_service == "DSL", np.random.normal(55, 10, N), np.random.normal(25, 8, N)),
)
monthly_charges = np.round(np.clip(base_charge, 18, 120), 2)
total_charges = np.round(monthly_charges * tenure_months + np.random.normal(0, 20, N), 2)
total_charges = np.clip(total_charges, 0, None)

# ---------------------------------------------------------
# 2. Churn logic — make it realistic, not random
#    Higher churn risk if: month-to-month contract, high monthly
#    charges, low tenure, fiber optic internet, electronic check
# ---------------------------------------------------------
churn_score = (
    (contract == "Month-to-month") * 0.35
    + (payment_method == "Electronic check") * 0.15
    + (internet_service == "Fiber optic") * 0.15
    + (tenure_months < 12) * 0.25
    + (monthly_charges > 80) * 0.10
    + np.random.normal(0, 0.15, N)
)
churn_prob = 1 / (1 + np.exp(-(churn_score - 0.5) * 4))  # sigmoid squash
churn = np.where(np.random.rand(N) < churn_prob, "Yes", "No")

# ---------------------------------------------------------
# 3. Assemble DataFrame
# ---------------------------------------------------------
df = pd.DataFrame({
    "CustomerID": customer_id,
    "Gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "TenureMonths": tenure_months,
    "PhoneService": phone_service,
    "InternetService": internet_service,
    "Contract": contract,
    "PaymentMethod": payment_method,
    "PaperlessBilling": paperless_billing,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn,
})

# ---------------------------------------------------------
# 4. Inject intentional data quality issues (for Day 2 cleaning practice)
# ---------------------------------------------------------

# a) Missing values in TotalCharges (as blank strings, like the real Telco dataset)
df["TotalCharges"] = df["TotalCharges"].astype(object)
missing_idx = np.random.choice(df.index, size=60, replace=False)
df.loc[missing_idx, "TotalCharges"] = ""

# b) Missing values in Gender
missing_gender_idx = np.random.choice(df.index, size=40, replace=False)
df.loc[missing_gender_idx, "Gender"] = np.nan

# c) Inconsistent casing / spacing in categorical text
inconsistent_idx = np.random.choice(df.index, size=100, replace=False)
df.loc[inconsistent_idx, "InternetService"] = df.loc[inconsistent_idx, "InternetService"].str.upper()

inconsistent_idx2 = np.random.choice(df.index, size=80, replace=False)
df.loc[inconsistent_idx2, "Contract"] = df.loc[inconsistent_idx2, "Contract"].apply(lambda x: f"  {x} ")

# d) Duplicate rows
dupes = df.sample(30, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

# e) A few negative/invalid MonthlyCharges (data entry errors)
neg_idx = np.random.choice(df.index, size=15, replace=False)
df.loc[neg_idx, "MonthlyCharges"] = -df.loc[neg_idx, "MonthlyCharges"]

# f) Shuffle rows so issues aren't clustered
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

# ---------------------------------------------------------
# 5. Save
# ---------------------------------------------------------
output_path = "/home/claude/project2/data/customer_churn_raw.csv"
df.to_csv(output_path, index=False)

print(f"Generated {len(df)} rows -> {output_path}")
print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
print("\nData quality issues injected:")
print(f"  - Missing TotalCharges: {(df['TotalCharges'] == '').sum()}")
print(f"  - Missing Gender: {df['Gender'].isna().sum()}")
print(f"  - Duplicate rows: {df.duplicated(subset=['CustomerID']).sum()}")
print(f"  - Negative MonthlyCharges: {(df['MonthlyCharges'] < 0).sum()}")
