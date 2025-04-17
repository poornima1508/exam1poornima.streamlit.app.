import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Streamlit page config
st.set_page_config(page_title="Automobile Data Cleaner", layout="wide")

# Load and preprocess the dataset
column_names = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", 
    "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", 
    "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", 
    "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", 
    "price"
]
df = pd.read_csv('auto.csv', header=None, names=column_names)

# App Title
st.title("🚗 Automobile Data Exploration & Transformation")

# Introduction
st.markdown("""
Welcome to the Automobile Data Cleaner App!  
Here we perform key preprocessing steps and visualize important trends to better understand the dataset.

---

### 📌 Tasks Covered:
- Handling missing values in the `stroke` column
- Converting `highway-mpg` to `highway-L/100km`
- Normalizing the `height` column
- Creating dummy variables for `aspiration`
- Visualizing distributions and relationships
""")

# Dataset Preview
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# === Question 1: Handle Missing Values in 'stroke' ===
st.markdown("### 1️⃣ Handling Missing Values in `stroke`")
df['stroke'] = df['stroke'].replace('?', np.nan).astype(float)
mean_stroke = df['stroke'].mean()
df['stroke'].fillna(mean_stroke, inplace=True)
st.success(f"Missing values in `stroke` filled with mean: **{mean_stroke:.2f}**")

# === Question 2: Convert 'highway-mpg' to L/100km ===
st.markdown("### 2️⃣ Convert `highway-mpg` ➡️ `highway-L/100km`")
df['highway-mpg'] = pd.to_numeric(df['highway-mpg'], errors='coerce')
df['highway-L/100km'] = 235 / df['highway-mpg']
st.success("Conversion complete. Here's a sample:")
st.dataframe(df[['highway-mpg', 'highway-L/100km']].head())

# === Question 3: Normalize the 'height' column ===
st.markdown("### 3️⃣ Normalize the `height` Column")
df['height'] = pd.to_numeric(df['height'], errors='coerce')
df['height'] = df['height'] / df['height'].max()
st.success("`height` column normalized to range 0–1.")
st.dataframe(df[['height']].head())

# === Question 4: Create Dummy Variables for 'aspiration' ===
st.markdown("### 4️⃣ Creating Dummy Variables for `aspiration`")
st.write("Unique values in `aspiration`:", df['aspiration'].unique())
aspiration_dummies = pd.get_dummies(df['aspiration'], prefix='aspiration')
df = pd.concat([df.drop('aspiration', axis=1), aspiration_dummies], axis=1)
st.success("Dummy variables created:")
st.dataframe(df.filter(like='aspiration_').head())

# === Final Data Summary ===
st.markdown("### 📊 Final Dataset Shape")
col1, col2 = st.columns(2)
col1.metric("🧾 Rows", df.shape[0])
col2.metric("🧾 Columns", df.shape[1])

# === Visualizations Section ===
st.markdown("---")
st.header("📈 Data Visualizations")

# Histogram of 'height'
st.subheader("📏 Distribution of Normalized Height")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.histplot(df['height'], kde=True, bins=30, color='skyblue', ax=ax1)
ax1.set_title("Histogram of Normalized Height", fontsize=14)
ax1.set_xlabel("Normalized Height")
st.pyplot(fig1)

# Scatter Plot: Horsepower vs Curb-weight
st.subheader("⚙️ Horsepower vs Curb-weight (Colored by Body Style)")
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

# Box Plot: Price by Body Style
st.subheader("💲 Price Distribution by Body Style")
df['price'] = pd.to_numeric(df['price'], errors='coerce')
fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x='body-style', y='price', palette='pastel', ax=ax3)
ax3.set_title("Box Plot of Price by Body Style", fontsize=14)
ax3.set_xlabel("Body Style")
ax3.set_ylabel("Price")
st.pyplot(fig3)

# Line Plot: Engine Size vs Price
st.subheader("📈 Engine Size vs Price Trend")
df['engine-size'] = pd.to_numeric(df['engine-size'], errors='coerce')
fig4 = px.line(
    df.sort_values('engine-size'), x='engine-size', y='price',
    title="Price vs Engine Size",
    labels={"engine-size": "Engine Size", "price": "Price"},
    markers=True,
    template="plotly_dark"
)
st.plotly_chart(fig4, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<center>🧠 Created with ❤️ by Poornima</center>", unsafe_allow_html=True)
