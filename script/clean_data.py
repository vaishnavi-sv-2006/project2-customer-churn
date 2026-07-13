"""
clean_data.py
-------------
Day 2, Part 1: Clean the raw customer churn dataset.

Fixes:
1. Missing TotalCharges (blank strings) -> convert to numeric, impute
2. Missing Gender values -> impute with mode (most common value)
3. Inconsistent casing/whitespace in InternetService and Contract
4. Duplicate rows -> removed
5. Negative MonthlyCharges -> data entry errors, converted to positive
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------
df = pd.read_csv("/home/claude/project2/data/customer_churn_raw.csv")
print(f"Raw shape: {df.shape}")

# ---------------------------------------------------------
# 2. Fix TotalCharges: blank strings -> NaN -> numeric -> impute
# ---------------------------------------------------------
# Blank strings can't be read as numbers, so first replace "" with NaN
df["TotalCharges"] = df["TotalCharges"].replace("", np.nan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Impute missing TotalCharges as MonthlyCharges * TenureMonths
# (a reasonable estimate, since TotalCharges should roughly equal that)
missing_total = df["TotalCharges"].isna()
df.loc[missing_total, "TotalCharges"] = (
    df.loc[missing_total, "MonthlyCharges"] * df.loc[missing_total, "TenureMonths"]
)
print(f"Fixed {missing_total.sum()} missing TotalCharges values")

# ---------------------------------------------------------
# 3. Fix missing Gender: impute with mode
# ---------------------------------------------------------
missing_gender = df["Gender"].isna().sum()
gender_mode = df["Gender"].mode()[0]
df["Gender"] = df["Gender"].fillna(gender_mode)
print(f"Fixed {missing_gender} missing Gender values (filled with '{gender_mode}')")

# ---------------------------------------------------------
# 4. Fix inconsistent text: casing + whitespace
# ---------------------------------------------------------
text_cols = ["Gender", "Partner", "Dependents", "PhoneService", "InternetService",
             "Contract", "PaymentMethod", "PaperlessBilling", "Churn"]
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()  # remove leading/trailing spaces
    # Standardize casing: title case for readability, but keep known formats consistent
df["InternetService"] = df["InternetService"].str.replace(
    "FIBER OPTIC", "Fiber optic", case=False
).str.title().replace({"No": "No"})  # keep "No" as-is
# Simpler, safer approach: use a mapping to normalize known categories
internet_map = {"DSL": "DSL", "FIBER OPTIC": "Fiber optic", "FIBER OPTIC ".strip(): "Fiber optic",
                "Fiber Optic": "Fiber optic", "No": "No"}
df["InternetService"] = df["InternetService"].str.upper().str.strip().map(
    {"DSL": "DSL", "FIBER OPTIC": "Fiber optic", "NO": "No"}
)

df["Contract"] = df["Contract"].str.strip()

print("Standardized text formatting in categorical columns")

# ---------------------------------------------------------
# 5. Remove duplicate rows (based on CustomerID)
# ---------------------------------------------------------
before = len(df)
df = df.drop_duplicates(subset=["CustomerID"], keep="first")
print(f"Removed {before - len(df)} duplicate rows")

# ---------------------------------------------------------
# 6. Fix negative MonthlyCharges (data entry errors -> take absolute value)
# ---------------------------------------------------------
negative_charges = (df["MonthlyCharges"] < 0).sum()
df["MonthlyCharges"] = df["MonthlyCharges"].abs()
print(f"Fixed {negative_charges} negative MonthlyCharges values")

# ---------------------------------------------------------
# 7. Final checks
# ---------------------------------------------------------
print(f"\nFinal shape: {df.shape}")
print(f"Remaining nulls:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ---------------------------------------------------------
# 8. Save cleaned dataset
# ---------------------------------------------------------
output_path = "/home/claude/project2/data/customer_churn_cleaned.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved cleaned data -> {output_path}")
