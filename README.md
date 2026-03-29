# 🚀 Azure Demand Forecasting & Capacity Optimization System
## 📊 End-to-End Azure Capacity Optimization & Demand Forecasting Project

An end-to-end data engineering, machine learning, and deployment project designed to analyze, clean, validate, forecast, and monitor Azure cloud demand data for intelligent capacity planning and decision support.

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
> Clean and validate cloud demand data  
> Engineer forecasting-ready features  
> Train and evaluate predictive models  
> Deploy an interactive dashboard for forecasting and capacity planning  
> Monitor risk, utilization, waste, and operational insights  

---

## 🌟 Key Features (Implemented)

✔ Clean real-world noisy cloud data  
✔ Validate business constraints  
✔ Analyze demand patterns  
✔ Prepare data for forecasting models  
✔ Train ARIMA, XGBoost, and LSTM models  
✔ Select the best-performing forecasting model  
✔ Deploy an interactive Streamlit dashboard  
✔ Generate real-time scenario-based demand forecasts  
✔ Monitor capacity risk, underutilization, and wasted cost  
✔ Export audit logs and dashboard reports  


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
│   ├── Milestone3_Model_Development.ipynb
│   ├── xgboost_demand_forecast_model.pkl
│   ├── model_evaluation_report.md
│   ├── agile_documentation.md
│   ├── Feature_list.pkl
│
├── milestone4_deployment/
|   ├── Milestone4_Forecast_Integration_CapacityPlanning.ipynb
│   ├── app.py
│   ├── streamlit_prediction_log.csv
│   ├── README_assets/
│   ├── Azure_Demand_Forecasting_demo_video.mp4
│
├── docs/
│   ├── Defect_tracker_Azure_Demand_Forecasting.xlsx
│   ├── Unit_Test_Plan_Azure_Demand_Forecasting.xlsx
│
│
├── data/
│   ├── raw_dataset.csv
    ├── feature_engineered_dataset.csv
│
├── Requirements.txt
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

## 🤖 Milestone 3 – Machine Learning Model Development

> Milestone 3 focuses on building predictive models capable of forecasting future Azure cloud demand using the engineered time-series dataset.

## 🎯 Objective
* Develop machine learning and statistical forecasting models
* Train models using historical demand signals
* Evaluate model performance using industry-standard metrics
* Identify the best-performing approach for capacity demand prediction

## 🧠 Models Implemented
### 🔹 ARIMA (AutoRegressive Integrated Moving Average)
> A classical statistical time-series forecasting model used to capture:
> Temporal trends
> Autocorrelation patterns
> Demand seasonality behavior
> ARIMA was trained on the aggregated demand time-series to establish a statistical forecasting baseline.

### 🔹 XGBoost Regression Model
> A powerful gradient boosting machine learning algorithm used for structured datasets.
> XGBoost leverages the engineered features from Milestone 2 including:
> Lag features
> Rolling statistics
> Business context features
> Encoded categorical variables
> This allows the model to capture complex nonlinear relationships in demand behavior.

### 🔹 LSTM (Long Short-Term Memory) Deep Learning Model
> LSTM is a recurrent neural network architecture designed specifically for sequential and time-series data.
> The LSTM model captures long-term temporal dependencies in historical demand signals and learns complex sequential patterns that traditional statistical models may miss.
> This deep learning approach enhances forecasting capability by modeling nonlinear time-based relationships within the demand data.

## ⚙️ Model Training Workflow

> The model development pipeline includes:

✔ Dataset loading from feature_engineered_dataset.csv
✔ Timestamp conversion and time ordering
✔ Feature / target variable separation
✔ Train-test split using time-series methodology
✔ Model training using ARIMA and XGBoost and LSTM
✔ Demand prediction generation
✔ Performance evaluation using regression metrics

## 📊 Model Evaluation Metrics
> To assess forecasting performance, the following metrics were used:
> MAE (Mean Absolute Error)
  * Measures the average absolute difference between predicted and actual demand.
> RMSE (Root Mean Squared Error)
  * Penalizes large forecasting errors and highlights model stability.
> R² Score (Coefficient of Determination)
  * Indicates how well the model explains variance in demand patterns.

## 📈 Model Performance Comparison
* The project compares statistical forecasting with machine learning methods to determine which approach better captures Azure infrastructure demand dynamics.
* Model	MAE	RMSE	R² Score
* ARIMA	Baseline Forecast	Baseline Error	Baseline Fit
* XGBoost	Improved Prediction	Lower Error	Higher Variance Explanation
* XGBoost demonstrates strong performance due to its ability to leverage engineered features and nonlinear relationships.
* After evaluation using MAE, RMSE, and R² metrics, XGBoost achieved the most accurate and stable predictions, making it the selected model for deployment in the next milestone.

| Model   | Type              | Performance |
|--------|------------------|------------|
| ARIMA  | Statistical Model | Baseline Forecast |
| XGBoost| Machine Learning  | Best Performance |
| LSTM   | Deep Learning     | Sequential Forecast |



## 📊 Feature Importance Analysis
* Feature importance analysis was conducted to understand the key drivers of Azure demand prediction.
* Important predictive signals include:
* Lag demand values
* Rolling demand averages
* Capacity utilization
* Market demand indicators
* Product launch impact
* This provides insights into the operational drivers influencing infrastructure demand.

### 📦 Output
* Trained forecasting models
* Model evaluation metrics
* Forecast predictions on test dataset
* Feature importance analysis
> These outputs establish a reliable demand prediction system ready for deployment in the next milestone.

## 🚀 Milestone 4 – Forecast Integration & Capacity Planning
> Milestone 4 focuses on transforming the trained forecasting model into a working decision-support dashboard for real-time planning, monitoring, and business insights.
## 🎯 Objective
* Deploy the selected forecasting model
* Integrate forecasting into an interactive Streamlit dashboard
* Provide scenario-based demand prediction
* Monitor utilization, waste cost, and regional risk
* Maintain forecast logs and reporting outputs
## 🧠 Milestone 4 Components
### 🔹 Model Deployment
* Loaded trained XGBoost forecasting model from .pkl
* Loaded saved feature schema from feature_list.pkl
* Created prediction-ready feature alignment logic
* Enabled scenario-based forecasting using user inputs
### 🔹 Dashboard Integration
> Built a professional Streamlit dashboard with multiple sections:
* KPI Overview
* Demand Trends
* Regional Analysis
* Model & Forecast
* Risk Alerts
### 🔹 Automation Features
Dynamic filtering by:
* Region
* Service Type
*  Year
*  Capacity Risk Threshold
  
Automated export options for:
*  Filtered dashboard data
*  Risk records
*  Summary report
*  Forecast audit log
### 🔹 Monitoring & Audit Logging
* Capacity risk events identified using threshold logic
* Underutilized records flagged for inefficiency detection
* Forecast results recorded in streamlit_prediction_log.csv
* Scenario comparison added for planning decisions
### 📊 Dashboard Highlights
KPI Overview
* Total Cost
* Wasted Capacity Cost
* Average Utilization
* Incidents
* Capacity Risk Events
* Underutilized Flags
* Average Headroom
* Average Daily Growth Rate

Demand Trends
* Monthly usage trend
* Growth rate trend
* Weekly seasonality
* Rolling demand statistics
* Utilization vs headroom by service
* Monthly waste cost trend
* Top 5 high-demand months

Regional Analysis
* Regional capacity risk bubble chart
* Top waste regions
* Top risk regions
* Region risk heatmap

Model & Forecast
* Scenario-based input controls
* Predicted usage gauge
* Short-term demand outlook
* What-if comparison table
* Decision recommendation logic

Risk Alerts
* High-risk record tracking
* Underutilization analysis
* Threshold-based risk monitoring
* Recent high-risk records table
### 📦 Output
* app.py
* Streamlit dashboard
* Real-time forecasting workflow
* Exportable reports
* Audit logs
* Final demo-ready deployed system

### 🔹 Milestone 4 – Deployment
* Streamlit dashboard
* Forecast visualization
* Real-time prediction interface
* Interactive cloud demand analytics

## 🌐 Live Deployment App

The project is successfully deployed on Streamlit Community Cloud.

🔗 **Live App:**  
[Azure Demand Forecasting & Capacity Optimization System](https://azure-demand-forecasting-capacity-optimization-system-fx2icjdr.streamlit.app/)

## 📌 Academic & Industry Value
✔ End-to-end ML pipeline thinking
✔ Real-world cloud infrastructure use case
✔ Business validation rules applied
✔ Forecast deployment and dashboard integration
✔ Monitoring and reporting capability
✔ Clean structured repository
✔ Scalable system architecture


## 👨‍💻 Author

Ashish Kumar Prusty
B.Tech – Artificial Intelligence & Machine Learning
GitHub: https://github.com/ASHISH8652

## 📜 License

This project is licensed under the MIT License.

“Data is not useful until it is clean, validated, trusted, and turned into decisions.”
