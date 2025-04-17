import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set Streamlit page configuration
st.set_page_config(layout="wide")

# Load the dataset
column_names = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", 
    "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", 
    "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", 
    "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", 
    "price"
]
df = pd.read_csv('auto.csv', header=None, names=column_names)

# App title
st.title('🚗 Automobile Data Exploration and Transformations')

# Introduction
st.markdown("""
Welcome! In this app, we explore and transform an automobile dataset. The key tasks include:

- 🔧 Handling missing values in the 'stroke' column  
- 🔄 Converting 'highway-mpg' to 'highway-L/100km'  
- 📏 Normalizing the 'height' column  
- 🎯 Creating dummy variables for the 'aspiration' column  
- 📊 Visualizing important trends and relationships in the data  
""")

# Dataset Preview
st.subheader("🔍 Dataset Overview")
st.dataframe(df.head())

# 1️⃣ Handle missing values in 'stroke'
st.subheader("1️⃣ Handling Missing Values in 'Stroke'")
df['stroke'] = df['stroke'].replace('?', np.nan).astype(float)
mean_stroke = df['stroke'].mean()
df['stroke'].fillna(mean_stroke, inplace=True)
st.write(f"✅ Replaced missing values in 'stroke' with the mean: **{mean_stroke:.2f}**")

# 2️⃣ Convert 'highway-mpg' to 'highway-L/100km'
st.subheader("2️⃣ Convert 'highway-mpg' ➡️ 'highway-L/100km'")
df['highway-mpg'] = pd.to_numeric(df['highway-mpg'], errors='coerce')
df['highway-L/100km'] = 235 / df['highway-mpg']
st.write("✅ Converted units. Here's a preview:")
st.dataframe(df[['highway-mpg', 'highway-L/100km']].head())

# 3️⃣ Normalize the 'height' column
st.subheader("3️⃣ Normalize the 'Height' Column")
df['height'] = pd.to_numeric(df['height'], errors='coerce')
df['height'] = df['height'] / df['height'].max()
st.write("✅ Normalized 'height' column. Here's a preview:")
st.dataframe(df[['height']].head())

# 4️⃣ Create dummy variables for 'aspiration'
st.subheader("4️⃣ Create Dummy Variables for 'Aspiration'")
st.write("Unique values found in `aspiration` column:", df['aspiration'].unique())
aspiration_dummies = pd.get_dummies(df['aspiration'], prefix='aspiration')
df = pd.concat([df, aspiration_dummies], axis=1)
df.drop('aspiration', axis=1, inplace=True)
st.write("✅ Created dummy variables. Preview:")
st.dataframe(df[['aspiration_std', 'aspiration_turbo']].head())

# 📊 Final dataset info
st.subheader("📊 Final Dataset Dimensions")
col1, col2 = st.columns(2)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])

# 📈 Visualizations
st.header("📈 Data Visualizations")

# Histogram of height
st.subheader("📏 Distribution of Normalized Height")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.histplot(df['height'], kde=True, bins=30, color='skyblue', ax=ax1)
ax1.set_title("Histogram of Normalized Height", fontsize=14)
ax1.set_xlabel("Normalized Height")
st.pyplot(fig1)

# Scatter plot: horsepower vs curb-weight
st.subheader("⚙️ Horsepower vs Curb-weight")
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['curb-weight'] = pd.to_numeric(df['curb-weight'], errors='coerce')
fig2 = px.scatter(
    df, x='horsepower', y='curb-weight',
    title="Horsepower vs Curb-weight",
    labels={"horsepower": "Horsepower", "curb-weight": "Curb Weight"},
    color='body-style',
    template="plotly_white"
)
st.plotly_chart(fig2, use_container_width=True)

# Box plot: Price by body style
st.subheader("💲 Price Distribution by Body Style")
df['price'] = pd.to_numeric(df['price'], errors='coerce')
order = df.groupby('body-style')['price'].mean().sort_values().index
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='body-style', y='price', order=order, palette='Set2', ax=ax3)
ax3.set_title("Box Plot of Price by Body Style", fontsize=16)
ax3.set_xlabel("Body Style", fontsize=12)
ax3.set_ylabel("Price", fontsize=12)
plt.xticks(rotation=45)
st.pyplot(fig3)

# Line plot: Price vs engine size (smoothed)
st.subheader("📈 Average Price vs Engine Size")
df['engine-size'] = pd.to_numeric(df['engine-size'], errors='coerce')
line_df = df[['engine-size', 'price']].dropna()
grouped = line_df.groupby('engine-size').mean().reset_index()
fig4 = px.line(
    grouped, x='engine-size', y='price',
    title="Average Price vs Engine Size",
    labels={"engine-size": "Engine Size", "price": "Average Price"},
    markers=True,
    template="plotly_white",
    line_shape='spline'
)
fig4.update_traces(line=dict(width=3), marker=dict(size=6))
fig4.update_layout(title_font_size=16)
st.plotly_chart(fig4, use_container_width=True)
