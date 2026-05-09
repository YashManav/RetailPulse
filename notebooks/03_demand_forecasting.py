#!/usr/bin/env python
# coding: utf-8

# # Module 3: Demand Forecasting
# Aggregate daily sales and use Prophet to forecast demand for the next 30 days.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import os

plt.style.use('ggplot')


# In[ ]:


if os.path.exists('../data/retail_data_clean.csv'):
    df = pd.read_csv('../data/retail_data_clean.csv', parse_dates=['InvoiceDate'])
else:
    print("Cleaned data not found. Please run 01_eda_cleaning.ipynb first.")
    # Dummy data
    dates = pd.date_range(start='2011-01-01', end='2011-12-31', freq='D')
    df = pd.DataFrame({
        'InvoiceDate': dates,
        'Sales': np.random.randint(100, 1000, size=len(dates))
    })


# In[ ]:


# Aggregate to daily sales
daily_sales = df.groupby(df['InvoiceDate'].dt.date)['Sales'].sum().reset_index()
daily_sales.columns = ['ds', 'y']
daily_sales['ds'] = pd.to_datetime(daily_sales['ds'])


# In[ ]:


# Train/Test Split (last 30 days for testing)
train_size = len(daily_sales) - 30
train = daily_sales.iloc[:train_size]
test = daily_sales.iloc[train_size:]

# Fit Prophet
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model.fit(train)


# In[ ]:


# Forecast
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Calculate MAPE on test set
forecast_test = forecast.iloc[-30:]['yhat'].values
actual_test = test['y'].values
mape = mean_absolute_percentage_error(actual_test, forecast_test)
print(f"MAPE on test set: {mape:.2%}")


# In[ ]:


# Plot forecast
fig = model.plot(forecast)
plt.title('30-Day Demand Forecast')
plt.show()


# In[ ]:


# Re-train on full data and forecast next 30 days
model_full = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model_full.fit(daily_sales)
future_full = model_full.make_future_dataframe(periods=30)
forecast_full = model_full.predict(future_full)

forecast_full[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_csv('../data/forecast_30d.csv', index=False)
print("Saved forecast to ../data/forecast_30d.csv")

