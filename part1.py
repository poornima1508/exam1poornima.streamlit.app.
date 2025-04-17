import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Automobile Data Analysis", layout="wide")

# Column names
column_names = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", 
    "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", 
    "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", 
    "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", 
    "price"
]

# Load data
df = pd.read_csv("auto.csv", header=None, names=column_names)
df.replace("?", np.nan, inplace=True)

# Convert columns to numeric where needed
for col in ['bore', 'stroke', 'compression-ratio', 'horsepower', 'price', 'peak-rpm', 'height', 'highway-mpg', 'engine-size', 'curb-weight']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Sidebar
st.sidebar.title("🚦 Navigation")
page = st.sidebar.radio("Go to", ["Part 1: Transformation", "Part 2: Analysis"])

# PART 1: Data Cleaning and Transformation
if page == "Part 1: Transformation":
    st.title("🔧 Part 1: Data Cleaning & Transformation")

    st.subheader("1️⃣ Handling Missing Values in `stroke`")
    mean_stroke = df['stroke'].mean()
    df['stroke'].fillna(mean_stroke, inplace=True)
    st.write(f"Filled missing 'stroke' values with mean: **{mean_stroke:.2f}**")

    st.subheader("2️⃣ Convert `highway-mpg` ➡️ `highway-L/100km`")
    df['highway-L/100km'] = 235 / df['highway-mpg']
    st.dataframe(df[['highway-mpg', 'highway-L/100km']].head())

    st.subheader("3️⃣ Normalize `height`")
    df['height_normalized'] = df['height'] / df['height'].max()
    st.dataframe(df[['height', 'height_normalized']].head())

    st.subheader("4️⃣ Dummy Variables for `aspiration`")
    df['aspiration'] = df['aspiration'].astype(str).str.strip()
    aspiration_dummies = pd.get_dummies(df['aspiration'], prefix='aspiration')
    df = pd.concat([df.drop(columns='aspiration'), aspiration_dummies], axis=1)
    st.dataframe(aspiration_dummies.head())

    st.markdown("### 📐 Dataset Dimensions")
    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.subheader("📊 Preview of Cleaned Dataset")
    st.dataframe(df.head())

# PART 2: Statistical Analysis and Visualizations
if page == "Part 2: Analysis":
    st.title("📈 Part 2: Statistical Analysis & Visualization")

    st.subheader("🔍 Data Type of `peak-rpm`")
    st.write(f"Data type of `peak-rpm`: `{df['peak-rpm'].dtype}`")

    st.subheader("📊 Correlation Matrix")
    selected = df[['bore', 'stroke', 'compression-ratio', 'horsepower']].dropna()
    corr_matrix = selected.corr()
    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm').format("{:.2f}"))

    fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

    st.subheader("📉 Stroke vs Price Correlation")
    stroke_price = df[['stroke', 'price']].dropna()
    corr_val = stroke_price.corr().iloc[0, 1]
    st.write(f"Correlation coefficient: **{corr_val:.3f}**")

    fig_stroke, ax_stroke = plt.subplots(figsize=(7, 5))
    sns.regplot(x='stroke', y='price', data=stroke_price, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
    ax_stroke.set_title("Stroke vs Price")
    st.pyplot(fig_stroke)

    st.subheader("💰 Average Price by Body Style")
    body_price = df[['body-style', 'price']].dropna()
    grouped_price = body_price.groupby("body-style")['price'].mean().reset_index().sort_values(by='price', ascending=False)
    st.dataframe(grouped_price.style.format({"price": "{:.2f}"}))

    fig_bar = px.bar(grouped_price, x='body-style', y='price',
                     title="Average Price by Body Style",
                     labels={'price': 'Average Price', 'body-style': 'Body Style'},
                     color='price', color_continuous_scale='Blues')
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("📈 Line Plot: Price vs Engine Size")
    line_data = df[['engine-size', 'price']].dropna().groupby('engine-size').mean().reset_index()
    fig_line = px.line(line_data, x='engine-size', y='price', markers=True,
                       title="Average Price by Engine Size", template='plotly_dark')
    st.plotly_chart(fig_line, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>👩‍💻 Built by Poornima | Streamlit App</div>", unsafe_allow_html=True)
