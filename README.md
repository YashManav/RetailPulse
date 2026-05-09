# RetailPulse

RetailPulse is an AI-powered retail analytics platform that provides insights into customer behavior, demand forecasting, customer churn, and inventory optimization.

## Modules

1. **EDA & Cleaning**: Data exploration, missing value handling, sales trends, and correlation analysis.
2. **Customer Segmentation**: RFM scoring and K-Means clustering.
3. **Demand Forecasting**: Daily sales aggregation and 30-day forecasting using Prophet.
4. **Churn Prediction**: Churn prediction using XGBoost and feature importance via SHAP.
5. **Inventory Optimization**: Reorder point computation based on forecasted demand.
6. **Streamlit Dashboard**: Interactive UI visualizing KPIs, segments, forecasts, and inventory.

## Setup Instructions

### 1. Download the Dataset
Download the [Online Retail II](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) dataset from Kaggle.
Extract and rename the file to `retail_data.csv`. Place it in the `data/` directory:
```
RetailPulse/
└── data/
    └── retail_data.csv
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Run the following command to install required packages:
```bash
pip install -r requirements.txt
```

### 3. Run Jupyter Notebooks
Navigate to the `notebooks/` directory and run each notebook in order (01 to 05) to generate the necessary processed data and models.
```bash
jupyter notebook
```

### 4. Start the Streamlit Dashboard
After running the notebooks (which process the data), you can start the Streamlit dashboard:
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` to view the dashboard.
