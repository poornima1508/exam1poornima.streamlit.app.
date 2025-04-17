import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Page Configuration
st.set_page_config(page_title="Automobile Data - Part 2", layout="centered")

# Title
st.title("🚗 Part 2: Statistical Analysis and Correlation")

# Load Dataset
column_names = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", 
    "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", 
    "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", 
    "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", 
    "price"
]
df = pd.read_csv("CleanedAutomobile.csv", header=None, names=column_names)

# Preprocessing
df.replace('?', np.nan, inplace=True)
numeric_cols = ['bore', 'stroke', 'compression-ratio', 'horsepower', 'price', 'peak-rpm']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

st.markdown("---")

# Q1: Data type of 'peak-rpm'
st.subheader("🔍 Q1: Data Type of 'peak-rpm'")
st.write(f"The data type of **'peak-rpm'** is: `{df['peak-rpm'].dtype}`")

st.markdown("---")

# Q2: Correlation Matrix
st.subheader("📊 Q2: Correlation Among 'bore', 'stroke', 'compression-ratio', and 'horsepower'")
correlation_matrix = df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()
st.write("Correlation Matrix:")
st.dataframe(correlation_matrix.style.background_gradient(cmap='coolwarm').format("{:.2f}"))

# Heatmap
fig1, ax1 = plt.subplots(figsize=(7, 5))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax1)
ax1.set_title("Correlation Heatmap", fontsize=14)
st.pyplot(fig1)

st.markdown("---")

# Q3a: Correlation between 'stroke' and 'price'
st.subheader("📈 Q3a: Correlation Between 'Stroke' and 'Price'")
corr_value = df[['stroke', 'price']].corr().loc['stroke', 'price']
st.write(f"The correlation coefficient between **stroke** and **price** is: `{corr_value:.3f}`")

st.markdown("---")

# Q3b: Scatter Plot with Regression Line
st.subheader("📉 Q3b: Relationship Between 'Stroke' and 'Price'")
df_clean = df[['stroke', 'price']].dropna()

fig2, ax2 = plt.subplots(figsize=(7, 5))
sns.regplot(x='stroke', y='price', data=df_clean, scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
ax2.set_title("Stroke vs Price with Regression Line", fontsize=14)
ax2.set_xlabel("Stroke")
ax2.set_ylabel("Price")
st.pyplot(fig2)

st.write("""
A weak or near-zero correlation suggests **no strong linear relationship** between stroke and price.
Visually, the data points are scattered, and the regression line is almost flat.
""")

st.markdown("---")

# Q4: Average Price by Body Style
st.subheader("💰 Q4: Average Price by Body Style")
avg_price = df.groupby('body-style')['price'].mean().reset_index().sort_values(by='price', ascending=False)
st.dataframe(avg_price.style.format({"price": "{:.2f}"}))

# Bar Chart
fig3 = px.bar(
    avg_price, 
    x='body-style', 
    y='price', 
    title="Average Car Price by Body Style",
    labels={"price": "Average Price", "body-style": "Body Style"},
    color='price', 
    color_continuous_scale="Blues"
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# Footer
st.markdown("<div style='text-align: center;'>Made with ❤️ by Poornima</div>", unsafe_allow_html=True)
