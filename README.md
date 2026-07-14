# Customer Churn Prediction

An end-to-end data analytics project that predicts customer churn for a subscription-based business using Python and visualizes key drivers in Power BI. This is Project 2 in a 4-part data analytics portfolio (Sales Trend Visualization → **Customer Churn Prediction** → Big Data Insights Dashboard → AI-Powered Business Intelligence Dashboard).

## 📌 Overview

Customer churn — when a customer cancels a subscription — is one of the most costly problems for subscription businesses. This project builds a full pipeline: data generation, cleaning, exploratory analysis, and a machine learning model to predict which customers are likely to churn, so a business could intervene early.

## 🗂️ Dataset

A synthetic dataset of 5,000+ customer records styled after the well-known Telco Customer Churn dataset, with realistic churn patterns (encoded via a probability model based on contract type, internet service, tenure, and billing) and intentional data quality issues for cleaning practice.

**Columns:** CustomerID, Gender, SeniorCitizen, Partner, Dependents, TenureMonths, PhoneService, InternetService, Contract, PaymentMethod, PaperlessBilling, MonthlyCharges, TotalCharges, Churn

## 🔧 Methodology

| Day | Task |
|---|---|
| 1 | Generated synthetic dataset with realistic churn patterns |
| 2 | Cleaned data (missing values, duplicates, formatting) + exploratory data analysis |
| 3 | Built and evaluated Logistic Regression and Random Forest models |
| 4 | Built Power BI dashboard and documented findings |

## 📊 Key Insights

- **Contract type is the strongest churn driver** — Month-to-month customers churn at ~52%, vs ~23-25% for annual contracts
- **Fiber optic internet subscribers** churn nearly 2x more than DSL/no-internet customers
- **Electronic check payers** churn more than customers on automatic payment methods
- Churned customers tend to have **shorter tenure** and **higher monthly charges**

## 🤖 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 64.9% | 57.7% | 46.9% | 51.7% |
| **Random Forest** | **65.7%** | 59.3% | 46.1% | 51.9% |

**Top predictive features:** Contract, MonthlyCharges, TotalCharges, TenureMonths, PaymentMethod

## 🛠️ Tech Stack

- **Python** — pandas, numpy, scikit-learn
- **Power BI Desktop** — dashboard visualization
- **GitHub** — version control & hosting

## 📁 Repository Structure

```
project2-customer-churn/
├── data/
│   ├── customer_churn_raw.csv
│   ├── customer_churn_cleaned.csv
│   ├── summary_churn_by_contract.csv
│   ├── summary_churn_by_internet.csv
│   ├── summary_churn_by_payment.csv
│   ├── summary_stats.csv
│   ├── model_comparison.csv
│   └── feature_importance.csv
├── script/
│   ├── generate_data.py
│   ├── clean_data.py
│   ├── eda.py
│   └── churn_model.py
├── dashboard/
│   ├── churn_dashboard.pbix
│   └── dashboard_screenshot.png
├── .gitignore
├── LICENSE
└── README.md
```

## ▶️ How to Run

1. Clone this repository
2. Install dependencies: `pip install pandas numpy scikit-learn`
3. Run scripts in order:
   ```
   python script/generate_data.py
   python script/clean_data.py
   python script/eda.py
   python script/churn_model.py
   ```
4. Open `dashboard/churn_dashboard.pbix` in Power BI Desktop to view the interactive dashboard

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
