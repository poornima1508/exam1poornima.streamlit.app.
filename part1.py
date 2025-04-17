import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit page setup
st.set_page_config(page_title="Automobile Analysis", layout="wide")

# Column headers
column_names = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", 
    "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", 
    "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", 
    "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", 
    "price"
]

# Load data
df = pd.read_csv("auto.csv", header=None, names=column_names)

# Clean selected columns
def clean_column(col):
    return pd.to_numeric(df[col].astype(str).str.strip().replace('?', np.nan), errors='coerce')

df['stroke'] = clean_column('stroke')
df['highway-mpg'] = clean_column('highway-mpg')
df['height'] = clean_column('height')
df['price'] = clean_column('price')
df['engine-size'] = clean_column('engine-size')
df['horsepower'] = clean_column('horsepower')
df['curb-weight'] = clean_column('curb-weight')
df['normalized-losses'] = clean_column('normalized-losses')  # ✅ FIX ADDED

# App title
st.title("🚗 Automobile Data Exploration App")

# --- Dataset Overview ---
st.header("📊 Dataset Preview")
st.dataframe(df.head())

# --- Stroke missing value handling ---
st.subheader("1️⃣ Handling Missing Values: Stroke")
mean_stroke = df['stroke'].mean()
df['stroke'].fillna(mean_stroke, inplace=True)
st.write(f"Mean value used to fill missing 'stroke': **{mean_stroke:.2f}**")

# --- Handling Missing Values: Normalized Losses ---
st.subheader("🛠 Handling Missing Values: Normalized Losses")
mean_losses = df['normalized-losses'].mean()
df['normalized-losses'].fillna(mean_losses, inplace=True)
st.write(f"Filled missing values in 'normalized-losses' with mean: **{mean_losses:.2f}**")

# --- Convert highway-mpg to L/100km ---
st.subheader("2️⃣ Converting `highway-mpg` ➡️ `highway-L/100km`")
df['highway-L/100km'] = 235 / df['highway-mpg']
st.dataframe(df[['highway-mpg', 'highway-L/100km']].head())

# --- Normalize height ---
st.subheader("3️⃣ Normalizing Height")
df['height_normalized'] = df['height'] / df['height'].max()
st.dataframe(df[['height', 'height_normalized']].head())

# --- Dummy variables for aspiration ---
st.subheader("4️⃣ Creating Dummy Variables for Aspiration")
df['aspiration'] = df['aspiration'].astype(str).str.strip()
aspiration_dummies = pd.get_dummies(df['aspiration'], prefix='aspiration')
df = pd.concat([df, aspiration_dummies], axis=1)
df.drop(columns=['aspiration'], inplace=True)
st.dataframe(aspiration_dummies.head())

# --- Final dataset shape ---
st.markdown("### 📐 Dataset Dimensions")
col1, col2 = st.columns(2)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])

# ========== VISUALIZATIONS ==========

st.header("📈 Visual Insights")

# 1. Histogram: Normalized Height
st.subheader("📏 Histogram of Normalized Height")
fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.histplot(df['height_normalized'], bins=30, kde=True, color='teal', ax=ax1)
ax1.set_title("Distribution of Normalized Height")
ax1.set_xlabel("Normalized Height")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# 2. Scatter plot: Horsepower vs Curb-weight
st.subheader("⚙️ Horsepower vs Curb-weight")
scatter_df = df[['horsepower', 'curb-weight', 'body-style']].dropna()
fig2 = px.scatter(
    scatter_df, x='horsepower', y='curb-weight', color='body-style',
    title="Horsepower vs Curb-weight by Body Style",
    template='plotly_white'
)
st.plotly_chart(fig2, use_container_width=True)

# 3. Box plot: Price by Body Style
st.subheader("💲 Price Distribution by Body Style")
box_df = df[['body-style', 'price']].dropna()
fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.boxplot(x='body-style', y='price', data=box_df, palette='pastel', ax=ax3)
ax3.set_title("Box Plot of Price by Body Style", fontsize=14)
ax3.set_xlabel("Body Style")
ax3.set_ylabel("Price")
st.pyplot(fig3)

# 4. Line plot: Price vs Engine Size
st.subheader("📈 Price Trend by Engine Size")
line_df = df[['engine-size', 'price']].dropna().groupby('engine-size').mean().reset_index()
fig4 = px.line(
    line_df, x='engine-size', y='price', markers=True,
    title="Average Price by Engine Size", template='plotly_dark'
)
st.plotly_chart(fig4, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("👩‍💻 Built by Poornima | Streamlit App for Data Exploration")
