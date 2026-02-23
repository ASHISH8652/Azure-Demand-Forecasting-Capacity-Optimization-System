# 🚀 Azure Demand Forecasting & Capacity Optimization System
## 📊 End-to-End Azure Capacity Optimization  & Demand Forecasting Project

An end-to-end data engineering and machine learning project designed to analyze, clean, validate, and prepare Azure cloud demand data for intelligent forecasting and capacity planning.
--
This project simulates a real-world cloud infrastructure analytics pipeline, following professional ML workflow practices across four milestones.
--

## 🌍 Overview

> Cloud service providers must forecast infrastructure demand accurately to:
> Prevent over-provisioning (wasted cost)
> Prevent under-provisioning (service outages)
> Maintain high availability
> Optimize operational efficiency
> This project builds a structured system to:
---
## 🌟 Key Features (Implemented)
✔ Clean real-world noisy cloud data

✔ Validate business constraints

✔ Analyze demand patterns

✔ Prepare data for forecasting models

✔ Eventually deploy a forecasting pipeline

---
---
## 🏗️ Project Architecture (Milestone-Based Development)
```
Azure-Demand-Forecasting-Capacity-Optimization-System/
│
├── milestone1_data_cleaning/
│   ├── notebook.ipynb
│   ├── cleaned_dataset.csv
│
├── milestone2_feature_engineering/
│   ├── Milestone2_Feature_Engineering.ipynb
│   ├── feature_engineered_dataset.csv
│
├── milestone3_model_training/
│
├── milestone4_deployment/
│
├── data/
│   ├── raw_dataset.csv
│
├── requirements.txt
├── LICENSE
└── README.md
```
---
## 🧹 Milestone 1 – Data Cleaning & Exploratory Data Analysis

* Milestone 1 focuses on transforming a noisy bi-weekly Azure dataset into a validated, production-ready dataset.
---
## 🔎 Dataset Features
* Column	Description 
* time_stamp	Bi-weekly usage date
* region	Azure deployment region
* service_type	Compute / Storage
* usage_units	Actual demand
* provisioned_capacity	Allocated capacity
* cost_usd	Usage cost
* availability_pct	Service uptime percentage
* ⚙️ Data Quality Issues Handled
---
### The raw dataset intentionally contained:

* Missing values (~5%)
* Duplicate records (~3%)
* Inconsistent region formatting
* Cost rounding inconsistencies
* Business rule violations
  
### 🛠 Cleaning Steps Implemented

✔ Duplicate removal
✔ Missing value imputation
✔ Datetime conversion
✔ Region standardization
✔ Time-series interpolation
✔ Cost rounding correction
✔ Business validation rules
✔ Final dataset verification

### 📈 Exploratory Data Analysis

## Milestone 1 includes:

📊 Overall demand trend over time
📊 Region-wise average demand
📊 Service-type specific demand trend
📊 Statistical summary validation

## 🧠 Technologies Used

* **Python 3.9+**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Jupyter Notebook**

## 📊 Business Validation Logic

### To ensure real-world correctness:
* Usage must not exceed provisioned capacity
* Availability must remain between 90% – 100%
* Cost values standardized to 2 decimal precision
* Time-series data properly formatted

## 🔧 Milestone 2 – Feature Engineering & Data Wrangling

### 🔬 Milestone 2 – Feature Engineering & Data Wrangling

> Milestone 2 transforms the cleaned dataset into a model-ready forecasting dataset by enriching it with time-series intelligence and business-driven derived features.

## 🎯 Objective
* Prepare the dataset for forecasting models through:
* Identification of demand-driving variables
* Creation of lag-based historical influence features
* Detection of abnormal usage spikes
* Engineering rolling statistics for trend smoothing
* Structuring consistent time-series schema

## 🧠 Feature Engineering Implemented
### 🔹 Time-Based Features
* Year
* Month
* Quarter
* Week of Year
* Month Start / End Flags

> These allow models to understand seasonal demand behavior.

### 🔹 Lag Features
* lag_1
* lag_2
* lag_4
* lag_8

> These capture historical demand memory across region + service combinations.

### 🔹 Rolling Statistics
* rolling_mean_3
* rolling_mean_6
* rolling_std_3
* rolling_std_6

> These smooth short-term fluctuations and measure volatility.

### 🔹 Business Context Features
* Capacity Utilization (usage / provisioned_capacity)
* Growth Rate (short-term & medium-term)
* Demand Spike Flag (statistical anomaly detection)

> These features connect technical modeling with business impact.

### 🔹 Data Wrangling Steps

✔ Time sorting per region + service
✔ Consistent time granularity
✔ Categorical encoding
✔ Removal of lag-induced null values
✔ Final model-ready schema export

### 📦 Output
* feature_engineered_dataset.csv

> This dataset is now ready for:
* ARIMA / SARIMA
* Prophet
* XGBoost
* LSTM


## 🚀 Upcoming Milestones
🔹 Milestone 3 – Model Development

* ARIMA / SARIMA
* LSTM
* Regression baselines
* Model evaluation metrics

🔹 Milestone 4 – Deployment

* Streamlit dashboard
* Forecast visualization
* Real-time prediction interface

## 📌 Academic & Industry Value

✔ End-to-end ML pipeline thinking
✔ Real-world cloud infrastructure use case
✔ Business validation rules applied
✔ Clean structured repository
✔ Scalable system architecture

## 👨‍💻 Author

Ashish Kumar Prusty
B.Tech – Artificial Intelligence & Machine Learning
GitHub: https://github.com/ASHISH8652

## 📜 License

This project is licensed under the MIT License.

“Data is not useful until it is clean, validated, and trusted.”
