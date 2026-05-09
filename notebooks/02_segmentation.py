#!/usr/bin/env python
# coding: utf-8

# # Module 2: Customer Segmentation
# Calculate RFM (Recency, Frequency, Monetary) scores and apply K-Means clustering.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

plt.style.use('ggplot')


# In[ ]:


if os.path.exists('../data/retail_data_clean.csv'):
    df = pd.read_csv('../data/retail_data_clean.csv', parse_dates=['InvoiceDate'])
else:
    print("Cleaned data not found. Please run 01_eda_cleaning.ipynb first.")
    # Fallback to dummy data
    df = pd.DataFrame({
        'InvoiceNo': ['1', '2', '3'], 'CustomerID': [101, 102, 101],
        'InvoiceDate': pd.to_datetime(['2011-01-01', '2011-06-01', '2011-12-01']),
        'Sales': [100, 200, 150]
    })


# In[ ]:


# Calculate RFM
# Set reference date to 1 day after the latest invoice date
ref_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (ref_date - x.max()).days, # Recency
    'InvoiceNo': 'nunique',                             # Frequency
    'Sales': 'sum'                                      # Monetary
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'Sales': 'Monetary'
}).reset_index()

rfm.head()


# In[ ]:


# Preprocess for K-Means (log transform + scaling)
features = ['Recency', 'Frequency', 'Monetary']
rfm_log = rfm[features].copy()
# Handle 0 values in Monetary by adding a small constant before log
rfm_log['Monetary'] = rfm_log['Monetary'].clip(lower=1)
rfm_log = np.log1p(rfm_log)

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)


# In[ ]:


# Apply K-Means with k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(rfm_scaled)
rfm['Cluster'] = kmeans.labels_

# Map clusters to human-readable labels (optional heuristic)
# Compute cluster means to assign labels
cluster_means = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
cluster_means['Score'] = cluster_means['Frequency'] + cluster_means['Monetary'] - cluster_means['Recency']
sorted_clusters = cluster_means.sort_values('Score').index

labels = {sorted_clusters[0]: 'Lost', sorted_clusters[1]: 'At Risk', sorted_clusters[2]: 'Loyal', sorted_clusters[3]: 'Champions'}
rfm['Segment'] = rfm['Cluster'].map(labels)

print(rfm['Segment'].value_counts())


# In[ ]:


# Save RFM Data
rfm.to_csv('../data/rfm_segments.csv', index=False)
print("Saved RFM data to ../data/rfm_segments.csv")


# In[ ]:


# Visualize Segments
plt.figure(figsize=(8,6))
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Segment', palette='Set1')
plt.title('Customer Segments: Recency vs Monetary')
plt.show()

