# 📈 Retail Sales Forecasting Dashboard

An end-to-end Machine Learning and Business Analytics project that forecasts retail sales, compares forecasting models, detects anomalies, segments products, and provides interactive business insights through a Streamlit dashboard.

## 🌐 Live Demo

🚀 **Try the application here:**

https://bhimishettilohith-retail-sales-forecasting-das-xylofyapp-ujca4m.streamlit.app/

## 💻 GitHub Repository

Source Code:

https://github.com/bhimishettilohith/Retail-Sales-Forecasting-Dashboard

---

# 📌 Project Overview

Retail businesses generate massive amounts of sales data every day. Making sense of this data is essential for improving inventory management, forecasting demand, and maximizing revenue.

This project uses Machine Learning and Business Analytics techniques to:

- Forecast future retail sales
- Compare multiple forecasting models
- Detect unusual sales patterns using anomaly detection
- Segment products based on their sales performance
- Visualize insights through an interactive Streamlit dashboard

The application enables businesses to make data-driven decisions with an easy-to-use interface.

---

# 🚀 Features

## 📊 Business Dashboard

- Total Sales KPI
- Total Orders
- Total Customers
- Average Sales
- Monthly Sales Trend
- Sales by Category
- Sales by Region
- Interactive Sales Explorer

---

## 📈 Forecast Explorer

Implemented forecasting using:

- SARIMA
- Prophet
- XGBoost (Best Performing Model)

Performance Evaluation Metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

Features:

- Historical Monthly Sales Trend
- Forecast Model Comparison
- Download Model Performance Report

---

## 🚨 Anomaly Detection

Implemented using **Isolation Forest**

Features:

- Weekly Sales Trend
- Highlighted Sales Anomalies
- Downloadable Anomaly Report
- Business Insights

---

## 📦 Product Segmentation

Implemented using:

- K-Means Clustering
- Principal Component Analysis (PCA)

Features:

- Product Cluster Visualization
- Cluster Summary
- Inventory Recommendations
- Download Cluster Report

---

## ℹ About

Displays:

- Dataset Summary
- Technologies Used
- Machine Learning Models
- Project Deliverables

---

# 🛠 Technologies Used

### Programming Language

- Python

### Framework

- Streamlit

### Libraries

- Pandas
- NumPy
- Plotly
- Scikit-Learn
- XGBoost

### Machine Learning Models

- SARIMA
- Prophet
- XGBoost
- Isolation Forest
- K-Means Clustering
- PCA

---

# 📂 Project Structure

```
Retail-Sales-Forecasting-Dashboard
│
└── Xylofy
    ├── app.py
    ├── requirements.txt
    ├── train.csv
    ├── processed_superstore.csv
    ├── monthly_sales_processed.csv
    ├── model_comparison.csv
    ├── anomaly_report.csv
    └── README.md
```

---

# 📊 Dataset

The project uses a retail sales dataset containing:

- Customer Information
- Product Information
- Product Categories
- Regions
- Order Details
- Shipping Information
- Sales Records

Additional processed datasets were generated for forecasting and analytics.

---

# 📈 Machine Learning Workflow

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Time Series Aggregation
5. Stationarity Testing (ADF Test)
6. Forecasting Models
7. Model Evaluation
8. Anomaly Detection
9. Product Segmentation
10. Interactive Dashboard Development

---

# 📊 Model Performance

| Model | MAE | RMSE | MAPE |
|--------|---------:|---------:|---------:|
| **XGBoost** | **14763.81** | **18337.41** | **14.48%** |
| SARIMA | 20275.64 | 21062.97 | 21.51% |
| Prophet | 20250.79 | 22318.41 | 21.86% |

🏆 **Best Performing Model: XGBoost**

---

# 📷 Dashboard Screenshots

## 📊 Dashboard

_Add Screenshot Here_

---

## 📈 Forecast Explorer

_Add Screenshot Here_

---

## 🚨 Anomaly Detection

_Add Screenshot Here_

---

## 📦 Product Segmentation

_Add Screenshot Here_

---

## ℹ About

_Add Screenshot Here_

---

# ▶ Installation

Clone the repository:

```bash
git clone https://github.com/bhimishettilohith/Retail-Sales-Forecasting-Dashboard.git
```

Navigate to the project folder:

```bash
cd Retail-Sales-Forecasting-Dashboard/Xylofy
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

# 💡 Business Insights

- XGBoost achieved the highest forecasting accuracy.
- Isolation Forest effectively detected unusual sales patterns.
- K-Means clustering grouped products based on demand characteristics.
- Interactive visualizations help businesses understand sales trends.
- The dashboard supports inventory planning and business decision-making.

---

# 🚀 Future Enhancements

- Real-time Sales Forecasting
- User Authentication
- Database Integration
- Power BI Dashboard Integration
- Interactive Geographical Maps
- Automated Model Retraining
- Export Dashboard Reports to PDF
- AI-powered Sales Insights

---

# 👨‍💻 Author

**Bhimishetti Lohith**

B.Tech – Computer Science Engineering (Artificial Intelligence & Machine Learning)

Sir Padampat Singhania University, Udaipur

GitHub:
https://github.com/bhimishettilohith

---

# 🙏 Acknowledgements

This project was developed as part of a Machine Learning and Business Analytics learning journey to demonstrate practical applications of data science techniques in retail sales forecasting.

---
