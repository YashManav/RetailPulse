#!/usr/bin/env python
# coding: utf-8

# # Module 5: Inventory Optimization
# Use forecasted demand to compute reorder points for top products.

# In[ ]:


import pandas as pd
import numpy as np
from prophet import Prophet
import os

import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[ ]:


if os.path.exists('../data/retail_data_clean.csv'):
    df = pd.read_csv('../data/retail_data_clean.csv', parse_dates=['InvoiceDate'])
else:
    print("Cleaned data not found. Using dummy data.")
    df = pd.DataFrame({
        'StockCode': ['A']*50 + ['B']*50,
        'Description': ['Product A']*50 + ['Product B']*50,
        'Quantity': np.random.randint(1, 20, 100),
        'InvoiceDate': pd.date_range('2011-01-01', periods=50, freq='D').tolist() * 2
    })


# In[ ]:


# Find top 10 products by quantity sold
top_products = df.groupby(['StockCode', 'Description'])['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
top_stock_codes = top_products['StockCode'].tolist()

print("Top Products for Inventory Optimization:")
print(top_products)


# In[ ]:


# Forecast demand for each top product and calculate reorder point
# Reorder Point = Lead Time Demand + Safety Stock
# Assume 7 days lead time
lead_time_days = 7
# Assume safety stock is 1.65 * std_dev of daily demand * sqrt(lead_time) (for 95% service level)
z_score = 1.65 

inventory_plan = []

for stock_code in top_stock_codes:
    product_df = df[df['StockCode'] == stock_code]
    desc = product_df['Description'].iloc[0]

    # Daily aggregation
    daily = product_df.groupby(product_df['InvoiceDate'].dt.date)['Quantity'].sum().reset_index()
    daily.columns = ['ds', 'y']
    daily['ds'] = pd.to_datetime(daily['ds'])

    if len(daily) < 10:
        continue # Skip products with too little history

    # Fit Prophet
    m = Prophet(daily_seasonality=False, yearly_seasonality=False)
    m.fit(daily)

    # Forecast next 7 days (lead time)
    future = m.make_future_dataframe(periods=lead_time_days)
    forecast = m.predict(future)

    # Expected demand during lead time
    lead_time_demand = forecast['yhat'].iloc[-lead_time_days:].sum()

    # Calculate safety stock based on historical std dev
    std_dev_demand = daily['y'].std()
    safety_stock = z_score * std_dev_demand * np.sqrt(lead_time_days)

    reorder_point = lead_time_demand + safety_stock

    inventory_plan.append({
        'StockCode': stock_code,
        'Description': desc,
        'LeadTimeDemand': max(0, lead_time_demand),
        'SafetyStock': max(0, safety_stock),
        'ReorderPoint': max(0, reorder_point)
    })

inventory_df = pd.DataFrame(inventory_plan)
inventory_df['ReorderPoint'] = np.ceil(inventory_df['ReorderPoint']).astype(int)
inventory_df['SafetyStock'] = np.ceil(inventory_df['SafetyStock']).astype(int)
inventory_df['LeadTimeDemand'] = np.ceil(inventory_df['LeadTimeDemand']).astype(int)

print("\nInventory Optimization Plan:")
print(inventory_df)


# In[ ]:


# Save inventory plan
inventory_df.to_csv('../data/inventory_plan.csv', index=False)
print("Saved inventory plan to ../data/inventory_plan.csv")

