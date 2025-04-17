import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set the page config for better layout
st.set_page_config(layout="wide")

# Load and preprocess the dataset
column_names = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", 
    "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", 
    "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", 
    "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", 
    "price"
]
df = pd.read_csv('auto.csv', header=None, names=column_names)

# Title of the app
st.title('🚗 Automobile Data Exploration and Transformations')

# Introduction text
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

# === Question 1: Handle Missing Values in 'stroke' Column ===
st.subheader("1️⃣ Handling Missing Values in 'Stroke'")
df['stroke'] = df['stroke'].replace('?', np.nan).astype(float)
mean_stroke = df['stroke'].mean()
df['stroke'].fillna(mean_stroke, inplace=True)
st.write(f"✅ Replaced missing values in 'stroke' with the mean: **{mean_stroke:.2f}**")

# === Question 2: Convert 'highway-mpg' to 'highway-L/100km' ===
st.subheader("2️⃣ Convert 'highway-mpg' ➡️ 'highway-L/100km'")
df['highway-L/100km'] = 235 / df['highway-mpg']
st.write("✅ Converted units. Here's a preview:")
st.dataframe(df[['highway-mpg', 'highway-L/100km']].head())

# === Question 3: Normalize the 'height' column ===
st.subheader("3️⃣ Normalize the 'Height' Column")
df['height'] = df['height'] / df['height'].max()
st.write("✅ Normalized 'height' column. Here's a preview:")
st.dataframe(df[['height']].head())

# === Question 4: Create Dummy Variables for 'aspiration' ===
st.subheader("4️⃣ Create Dummy Variables for 'Aspiration'")
st.write("Unique values found in `aspiration` column:", df['aspiration'].unique())
aspiration_dummies = pd.get_dummies(df['aspiration'], prefix='aspiration')
df = pd.concat([df, aspiration_dummies], axis=1)
df.drop('aspiration', axis=1, inplace=True)
st.write("✅ Created dummy variables. Preview:")
st.dataframe(df[['aspiration_std', 'aspiration_turbo']].head())

# === Final Data Summary ===
st.subheader("📊 Final Dataset Dimensions")
col1, col2 = st.columns(2)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])

# === Visualizations ===
st.header("📈 Data Visualizations")

# --- Histogram of 'height' ---
st.subheader("📏 Distribution of Normalized Height")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.histplot(df['height'], kde=True, bins=30, color='skyblue', ax=ax1)
ax1.set_title("Histogram of Normalized Height", fontsize=14)
ax1.set_xlabel("Normalized Height")
st.pyplot(fig1)

# --- Scatter Plot: Horsepower vs Curb-weight ---
st.subheader("⚙️ Horsepower vs Curb-weight")
fig2 = px.scatter(
    df, x='horsepower', y='curb-weight',
    title="Horsepower vs Curb-weight",
    labels={"horsepower": "Horsepower", "curb-weight": "Curb Weight"},
    color='body-style',
    template="plotly_white"
)
st.plotly_chart(fig2, use_container_width=True)

# --- Box Plot: Price by Body Style ---
st.subheader("💲 Price Distribution by Body Style")
fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x='body-style', y='price', palette='pastel', ax=ax3)
ax3.set_title("Box Plot of Price by Body Style", fontsize=14)
ax3.set_xlabel("Body Style")
ax3.set_ylabel("Price")
st.pyplot(fig3)

# --- Line Plot: Price vs Engine Size ---
st.subheader("📈 Price Trend by Engine Size")
fig4 = px.line(
    df.sort_values('engine-size'), x='engine-size', y='price',
    title="Price vs Engine Size",
    labels={"engine-size": "Engine Size", "price": "Price"},
    markers=True,
    template="plotly_dark"
)
st.plotly_chart(fig4, use_container_width=True)
