import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import datetime as dt

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("Customer Segmentation Dashboard")

# ---------------------------
# Load dataset
# ---------------------------
@st.cache_data
def load_data():
    # Get project root dynamically
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'Online Retail.xlsx')
    
    # Load Excel file
    df = pd.read_excel(DATA_PATH)
    
    # Clean data
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df = df[df['Country'] == 'United Kingdom']
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

df = load_data()

# ---------------------------
# RFM calculation
# ---------------------------
latest_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',                                 # Frequency
    'TotalPrice': 'sum'                                     # Monetary
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# ---------------------------
# Scale features
# ---------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# ---------------------------
# Sidebar - Cluster selection
# ---------------------------
st.sidebar.header("Cluster Settings")
k = st.sidebar.slider("Select number of clusters", min_value=2, max_value=10, value=4)

# ---------------------------
# K-Means clustering
# ---------------------------
kmeans = KMeans(n_clusters=k, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

selected_clusters = st.sidebar.multiselect(
    "Select clusters to display",
    options=rfm['Cluster'].unique(),
    default=rfm['Cluster'].unique()
)

filtered_data = rfm[rfm['Cluster'].isin(selected_clusters)]

# ---------------------------
# Cluster Summary Table
# ---------------------------
st.subheader("Cluster Summary")
st.dataframe(filtered_data.groupby('Cluster').mean().round(2))

# ---------------------------
# Frequency vs Monetary scatter
# ---------------------------
st.subheader("Customer Segments (Frequency vs Monetary)")
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=filtered_data,
    x='Frequency',
    y='Monetary',
    hue='Cluster',
    palette='Set2',
    s=100
)
plt.xlabel("Frequency")
plt.ylabel("Monetary")
plt.title("Customer Segments")
st.pyplot(plt)

# ---------------------------
# Recency Boxplot
# ---------------------------
st.subheader("Recency Distribution by Cluster")
plt.figure(figsize=(10,6))
sns.boxplot(data=filtered_data, x='Cluster', y='Recency', palette='Set2')
plt.title("Recency Distribution per Cluster")
st.pyplot(plt)
