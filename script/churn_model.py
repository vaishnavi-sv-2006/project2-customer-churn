"""
churn_model.py
--------------
Day 3: Build and evaluate churn prediction models.

Steps:
1. Load cleaned data
2. Encode categorical variables (models need numbers, not text)
3. Split into train/test sets
4. Train Logistic Regression and Random Forest
5. Evaluate both with Accuracy, Precision, Recall, F1
6. Extract feature importance from Random Forest
7. Save results for the portfolio (model_comparison.csv, feature_importance.csv)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------------------------------------------------
# 1. Load cleaned data
# ---------------------------------------------------------
df = pd.read_csv("/home/claude/project2/data/customer_churn_cleaned.csv")

# ---------------------------------------------------------
# 2. Encode categorical variables
# ---------------------------------------------------------
# Drop CustomerID - it's just an identifier, not a useful predictor
model_df = df.drop(columns=["CustomerID"])

# Target variable: Churn (Yes/No) -> 1/0
model_df["Churn"] = model_df["Churn"].map({"Yes": 1, "No": 0})

# Encode all remaining text columns using LabelEncoder
categorical_cols = model_df.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col])
    encoders[col] = le

print(f"Encoded columns: {categorical_cols}")

# ---------------------------------------------------------
# 3. Train/test split (80% train, 20% test)
# ---------------------------------------------------------
X = model_df.drop(columns=["Churn"])
y = model_df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ---------------------------------------------------------
# 4. Train models
# ---------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. Evaluate both models
# ---------------------------------------------------------
def evaluate(model, name):
    preds = model.predict(X_test)
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, preds), 3),
        "Precision": round(precision_score(y_test, preds), 3),
        "Recall": round(recall_score(y_test, preds), 3),
        "F1_Score": round(f1_score(y_test, preds), 3),
    }

results = [
    evaluate(log_reg, "Logistic Regression"),
    evaluate(rf, "Random Forest"),
]
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

results_df.to_csv("/home/claude/project2/data/model_comparison.csv", index=False)

# ---------------------------------------------------------
# 6. Feature importance (from Random Forest)
# ---------------------------------------------------------
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print("\nTop 5 most important features for predicting churn:")
print(importance_df.head(5).to_string(index=False))

importance_df.to_csv("/home/claude/project2/data/feature_importance.csv", index=False)

# ---------------------------------------------------------
# 7. Confusion matrix for the better model (Random Forest)
# ---------------------------------------------------------
cm = confusion_matrix(y_test, rf.predict(X_test))
print(f"\nConfusion Matrix (Random Forest):")
print(f"                  Predicted No   Predicted Yes")
print(f"Actual No         {cm[0][0]:<14}{cm[0][1]}")
print(f"Actual Yes        {cm[1][0]:<14}{cm[1][1]}")

print("\nSaved: model_comparison.csv, feature_importance.csv")
