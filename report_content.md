# RetailPulse Project Report

## 1. Executive Summary
RetailPulse is an AI-powered retail analytics platform designed to extract actionable insights from historical transactional data. This project leverages the Online Retail II dataset to demonstrate the application of machine learning techniques in solving real-world retail challenges. The platform features four core modules: Customer Segmentation, Demand Forecasting, Churn Prediction, and Inventory Optimization. These insights are made accessible through an interactive Streamlit dashboard.

## 2. Objective
The primary objective of this project is to build an end-to-end data science pipeline that:
- Cleans and prepares raw transactional data.
- Groups customers into distinct segments to tailor marketing strategies.
- Forecasts future product demand to optimize supply chain operations.
- Predicts customer churn to proactively retain high-value clients.
- Computes inventory reorder points to prevent stockouts and minimize holding costs.
- Provides a user-friendly dashboard to visualize Key Performance Indicators (KPIs) and model outputs.

## 3. Methodology

### 3.1 Exploratory Data Analysis & Data Cleaning
- **Data Loading:** Handled the Online Retail II dataset, treating the 'InvoiceDate' as a datetime object and converting IDs to appropriate types.
- **Cleaning:** Dropped rows with missing `CustomerID` to ensure robust customer-level analysis. Removed cancelled orders (invoices starting with 'C') and records with negative quantities.
- **Feature Engineering:** Calculated the `Sales` revenue by multiplying `Quantity` and `UnitPrice`.

### 3.2 Customer Segmentation (K-Means & RFM)
- **RFM Analysis:** Computed Recency (days since last purchase), Frequency (number of purchases), and Monetary (total spend) for each customer.
- **Clustering:** Applied K-Means clustering algorithm ($k=4$) on log-transformed and standardized RFM scores.
- **Segment Labeling:** Customers were grouped into four logical segments: Champions, Loyal, At Risk, and Lost, based on their aggregate RFM scores.

### 3.3 Demand Forecasting (Prophet)
- **Time Series Aggregation:** Aggregated the transactional data into daily sales figures.
- **Modeling:** Utilized Facebook Prophet to model daily sales, capturing weekly and yearly seasonality trends.
- **Evaluation:** Evaluated the model using Mean Absolute Percentage Error (MAPE) on a 30-day holdout set before predicting the next 30 days of future demand.

### 3.4 Churn Prediction (XGBoost & SHAP)
- **Labeling Churn:** Defined churn as a customer who has been inactive for the last 90 days of the observation period.
- **Classification Model:** Trained an XGBoost classifier using historical RFM features and total quantities to predict the probability of future churn.
- **Interpretability:** Leveraged SHAP (SHapley Additive exPlanations) values to identify which features most heavily influenced the model's churn predictions, typically finding Recency to be the most critical factor.

### 3.5 Inventory Optimization
- **Reorder Points Calculation:** Focused on the top 10 best-selling products. Used Prophet to forecast their demand over a 7-day lead time.
- **Safety Stock:** Computed safety stock using a 95% service level assumption based on the historical standard deviation of daily sales.
- **Final Metric:** Reorder Point = Lead Time Demand + Safety Stock.

## 4. Results & Findings

### 4.1 EDA & Segmentation
*(Insert Screenshot of Monthly Sales Trend here)*
- The data revealed distinct shopping patterns and seasonality, typical of retail environments.
- **Segmentation:** The K-Means model successfully separated the customer base. The 'Champions' segment, while smaller in number, contributed disproportionately to the total revenue.

### 4.2 Forecasting Accuracy
*(Insert Screenshot of Prophet Forecast Chart here)*
- The Prophet model was able to capture the overarching trends and weekly fluctuations, providing a realistic 30-day forecast.

### 4.3 Churn Risk
*(Insert Screenshot of SHAP Summary Plot here)*
- The XGBoost model achieved a strong ROC-AUC score.
- The SHAP analysis indicated that customers with high Recency (long time since last purchase) and low Frequency were at the highest risk of churning.

### 4.4 Inventory Plan
*(Insert Screenshot of Inventory Dashboard Table here)*
- By utilizing dynamic reorder points based on predicted demand rather than static rules, the platform suggests an inventory strategy that balances stock availability and holding costs.

## 5. Conclusion
The RetailPulse project successfully demonstrates a full-stack data science workflow. By combining robust machine learning algorithms (XGBoost, K-Means, Prophet) with an interactive Streamlit interface, it delivers actionable intelligence that retail businesses can use to drive growth, retain customers, and optimize operations.

---
**Repository:** [GitHub Link Placeholder]
**Video Demo:** [Video Link Placeholder]
