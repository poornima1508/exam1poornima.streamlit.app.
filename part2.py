import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set the title of the app
st.title("Part 2: Statistical Analysis & Correlation")

# Load the dataset
column_names = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", 
    "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", 
    "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", 
    "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", 
    "price"
]
df = pd.read_csv("CleanedAutomobile.csv", header=None, names=column_names)

# Replace '?' with NaN and convert relevant columns to numeric
df.replace('?', np.nan, inplace=True)
numeric_cols = ['bore', 'stroke', 'compression-ratio', 'horsepower', 'price', 'peak-rpm']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# -----------------------------------------------
# Question 1: Data type of 'peak-rpm'
st.subheader("Q1: Data Type of 'peak-rpm'")
st.write(f"Data type of 'peak-rpm': `{df['peak-rpm'].dtype}`")

# -----------------------------------------------
# Question 2: Correlation Matrix
st.subheader("Q2: Correlation Between 'bore', 'stroke', 'compression-ratio', and 'horsepower'")
correlation_matrix = df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()
st.write(correlation_matrix)

# Plot heatmap
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax1)
st.pyplot(fig1)

# -----------------------------------------------
# Question 3a: Correlation between 'stroke' and 'price'
st.subheader("Q3a: Correlation Between 'Stroke' and 'Price'")
corr_stroke_price = df[['stroke', 'price']].corr()
st.write(corr_stroke_price)

# -----------------------------------------------
# Question 3b: Visualizing Linear Relationship
st.subheader("Q3b: Linear Relationship Between 'Stroke' and 'Price'")
df_clean = df[['stroke', 'price']].dropna()
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.regplot(x='stroke', y='price', data=df_clean, ax=ax2, scatter_kws={"alpha":0.6})
ax2.set_title("Relationship Between Stroke and Price")
ax2.set_xlabel("Stroke")
ax2.set_ylabel("Price")
st.pyplot(fig2)

# -----------------------------------------------
# Question 4: Average price by body-style
st.subheader("Q4: Average Price by Body Style")
avg_price = df.groupby('body-style')['price'].mean().reset_index().sort_values(by='price', ascending=False)
st.dataframe(avg_price)

# Bar plot of average price by body style
fig3 = px.bar(avg_price, x='body-style', y='price', title="Average Price by Body Style", color='price')
st.plotly_chart(fig3)
