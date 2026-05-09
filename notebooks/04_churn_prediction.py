#!/usr/bin/env python
# coding: utf-8

# # Module 4: Churn Prediction
# Define churn (90-day inactivity), build an XGBoost classifier, and analyze feature importance via SHAP.

# In[ ]:


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import shap
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot')


# In[ ]:


if os.path.exists('../data/retail_data_clean.csv'):
    df = pd.read_csv('../data/retail_data_clean.csv', parse_dates=['InvoiceDate'])
else:
    print("Cleaned data not found. Using dummy data.")
    df = pd.DataFrame({
        'InvoiceNo': list(range(100)),
        'CustomerID': np.random.randint(1000, 1050, 100),
        'InvoiceDate': pd.date_range('2011-01-01', periods=100, freq='3D'),
        'Quantity': np.random.randint(1, 10, 100),
        'Sales': np.random.uniform(10, 100, 100)
    })


# In[ ]:


# Feature Engineering for Churn
# Determine observation point (e.g., 90 days before the max date)
max_date = df['InvoiceDate'].max()
observation_date = max_date - pd.Timedelta(days=90)

# Features are calculated on data BEFORE observation_date
hist_df = df[df['InvoiceDate'] <= observation_date]
# Target (churn = 1 if no purchases AFTER observation_date)
future_df = df[df['InvoiceDate'] > observation_date]
active_customers = future_df['CustomerID'].unique()

# Create feature set
features = hist_df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (observation_date - x.max()).days, # Recency at observation date
    'InvoiceNo': 'nunique', # Frequency
    'Sales': ['sum', 'mean'], # Monetary & AOV
    'Quantity': 'sum'
}).reset_index()
features.columns = ['CustomerID', 'Recency', 'Frequency', 'TotalSales', 'AvgOrderValue', 'TotalQuantity']

# Create target variable
features['Churn'] = (~features['CustomerID'].isin(active_customers)).astype(int)
print("Churn Rate:", features['Churn'].mean())


# In[ ]:


# Train Test Split
X = features.drop(['CustomerID', 'Churn'], axis=1)
y = features['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)


# In[ ]:


# Evaluation
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, preds))
print(f"ROC AUC: {roc_auc_score(y_test, probs):.4f}")


# In[ ]:


# SHAP Values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Feature Importance for Churn")
plt.tight_layout()
plt.show()


# In[ ]:


# Save data for dashboard
features['ChurnProb'] = model.predict_proba(X)[:, 1]
features.to_csv('../data/churn_predictions.csv', index=False)
print("Saved churn predictions to ../data/churn_predictions.csv")

