#!/usr/bin/env python
# coding: utf-8

# # Module 1: EDA & Data Cleaning
# This notebook covers loading the data, handling missing values, calculating sales, and visualizing trends.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('ggplot')
sns.set_palette('viridis')


# In[ ]:


# Load dataset
data_path = '../data/retail_data.csv'
if os.path.exists(data_path):
    df = pd.read_csv(data_path, encoding='ISO-8859-1')
    df.rename(columns={'Invoice': 'InvoiceNo', 'Price': 'UnitPrice', 'Customer ID': 'CustomerID'}, inplace=True)
    print("Dataset shape:", df.shape)
else:
    print("Dataset not found at '../data/retail_data.csv'. Please download it from Kaggle.")
    # Dummy data for demonstration
    df = pd.DataFrame({
        'InvoiceNo': ['536365', '536365', 'C536379', '536366'],
        'StockCode': ['85123A', '71053', 'D', '22633'],
        'Description': ['HEART T-LIGHT HOLDER', 'METAL LANTERN', 'Discount', 'HAND WARMER'],
        'Quantity': [6, 6, -1, 10],
        'InvoiceDate': ['12/1/2010 8:26', '12/1/2010 8:26', '12/1/2010 9:41', '12/2/2010 10:00'],
        'UnitPrice': [2.55, 3.39, 27.50, 1.85],
        'CustomerID': [17850.0, 17850.0, 14527.0, 17850.0],
        'Country': ['United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom']
    })


# In[ ]:


# Data Cleaning
# 1. Drop missing CustomerIDs
df_clean = df.dropna(subset=['CustomerID']).copy()
df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)

# 2. Convert InvoiceDate to datetime
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

# 3. Remove cancelled orders (InvoiceNo starts with 'C') and negative quantities
df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
df_clean = df_clean[df_clean['Quantity'] > 0]

# 4. Calculate Sales Revenue
df_clean['Sales'] = df_clean['Quantity'] * df_clean['UnitPrice']

print("Cleaned dataset shape:", df_clean.shape)


# In[ ]:


# Save cleaned dataset for downstream tasks
os.makedirs('../data', exist_ok=True)
df_clean.to_csv('../data/retail_data_clean.csv', index=False)
print("Saved cleaned data to ../data/retail_data_clean.csv")


# ## Exploratory Data Analysis

# In[ ]:


# Top 10 Countries by Sales
country_sales = df_clean.groupby('Country')['Sales'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=country_sales.values, y=country_sales.index)
plt.title('Top 10 Countries by Total Sales')
plt.xlabel('Total Sales')
plt.ylabel('Country')
plt.tight_layout()
plt.show()


# In[ ]:


# Monthly Sales Trend
df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M')
monthly_sales = df_clean.groupby('YearMonth')['Sales'].sum()

plt.figure(figsize=(12,5))
monthly_sales.plot(marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:


# Correlation Heatmap
# Select numerical columns
num_cols = df_clean[['Quantity', 'UnitPrice', 'Sales']]
corr = num_cols.corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

