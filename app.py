import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="RetailPulse Analytics", layout="wide")

# Utility function to load data
@st.cache_data
def load_data(filename):
    path = os.path.join('data', filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

st.sidebar.title("RetailPulse")
st.sidebar.markdown("AI-Powered Retail Analytics Platform")
page = st.sidebar.radio("Navigation", ["Overview & KPIs", "Customer Segmentation", "Demand Forecasting", "Churn & Inventory"])

# Try loading all necessary datasets
df_clean = load_data('retail_data_clean.csv')
df_rfm = load_data('rfm_segments.csv')
df_forecast = load_data('forecast_30d.csv')
df_churn = load_data('churn_predictions.csv')
df_inv = load_data('inventory_plan.csv')

if page == "Overview & KPIs":
    st.title("📊 Overview & KPIs")
    
    if df_clean is not None:
        # Calculate KPIs
        total_sales = df_clean['Sales'].sum()
        total_orders = df_clean['InvoiceNo'].nunique()
        total_customers = df_clean['CustomerID'].nunique()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sales (£)", f"£{total_sales:,.2f}")
        col2.metric("Total Orders", f"{total_orders:,}")
        col3.metric("Total Customers", f"{total_customers:,}")
        
        st.markdown("---")
        st.subheader("Sales Trend (Daily)")
        
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
        daily_sales = df_clean.groupby(df_clean['InvoiceDate'].dt.date)['Sales'].sum().reset_index()
        fig_trend = px.line(daily_sales, x='InvoiceDate', y='Sales', title='Daily Sales Over Time')
        st.plotly_chart(fig_trend, use_container_width=True)
        
        st.subheader("Top 10 Products by Revenue")
        top_products = df_clean.groupby('Description')['Sales'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_prod = px.bar(top_products, x='Sales', y='Description', orientation='h', title='Top Products')
        fig_prod.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_prod, use_container_width=True)
    else:
        st.warning("Cleaned data not found. Please run the EDA notebook first.")

elif page == "Customer Segmentation":
    st.title("🎯 Customer Segmentation")
    
    if df_rfm is not None:
        st.markdown("Customers are segmented into four groups using K-Means clustering on their Recency, Frequency, and Monetary (RFM) scores.")
        
        # Segment distribution
        segment_counts = df_rfm['Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Segment Distribution")
            fig_pie = px.pie(segment_counts, names='Segment', values='Count', hole=0.4, 
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            st.subheader("Recency vs Monetary by Segment")
            fig_scatter = px.scatter(df_rfm, x='Recency', y='Monetary', color='Segment',
                                     hover_data=['Frequency'], opacity=0.7)
            # Log scale for monetary due to skewness
            fig_scatter.update_yaxes(type="log")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.subheader("Segment Profile Averages")
        avg_profile = df_rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
        st.dataframe(avg_profile, use_container_width=True)
    else:
        st.warning("RFM data not found. Please run the Segmentation notebook first.")

elif page == "Demand Forecasting":
    st.title("📈 Demand Forecasting")
    
    if df_forecast is not None:
        st.markdown("30-Day future sales forecast generated using Facebook Prophet.")
        
        df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])
        
        fig = go.Figure()
        
        # Add prediction line
        fig.add_trace(go.Scatter(x=df_forecast['ds'], y=df_forecast['yhat'], 
                                 mode='lines', name='Forecast', line=dict(color='blue')))
        
        # Add uncertainty intervals
        fig.add_trace(go.Scatter(x=df_forecast['ds'].tolist() + df_forecast['ds'].tolist()[::-1],
                                 y=df_forecast['yhat_upper'].tolist() + df_forecast['yhat_lower'].tolist()[::-1],
                                 fill='toself', fillcolor='rgba(0,100,250,0.2)', line=dict(color='rgba(255,255,255,0)'),
                                 hoverinfo="skip", showlegend=True, name='Uncertainty Interval'))
        
        fig.update_layout(title="Next 30 Days Forecast", xaxis_title="Date", yaxis_title="Predicted Sales")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Forecasted Values Data")
        st.dataframe(df_forecast.set_index('ds').round(2), use_container_width=True)
    else:
        st.warning("Forecast data not found. Please run the Forecasting notebook first.")

elif page == "Churn & Inventory":
    st.title("⚠️ Churn Prediction & Inventory Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("At-Risk Customers (High Churn Probability)")
        if df_churn is not None:
            # Show top 20 customers with highest churn probability
            high_churn = df_churn.sort_values('ChurnProb', ascending=False).head(20)
            high_churn['ChurnProb'] = (high_churn['ChurnProb'] * 100).round(2).astype(str) + '%'
            st.dataframe(high_churn[['CustomerID', 'TotalSales', 'Recency', 'ChurnProb']], use_container_width=True)
        else:
            st.warning("Churn data not found.")
            
    with col2:
        st.subheader("Inventory Reorder Plan")
        if df_inv is not None:
            st.markdown("Reorder points for top 10 products based on forecasted lead-time demand.")
            # Highlight products that might need immediate reorder (dummy logic here)
            st.dataframe(df_inv, use_container_width=True)
        else:
            st.warning("Inventory data not found.")
