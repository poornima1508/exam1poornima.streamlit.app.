print("Part -1")

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

# Clean and convert numeric columns
def clean_column(col):
    return pd.to_numeric(df[col].astype(str).str.strip().replace('?', np.nan), errors='coerce')

numeric_columns = ['normalized-losses', 'stroke', 'highway-mpg', 'height', 'price', 
                   'engine-size', 'horsepower', 'curb-weight']
for col in numeric_columns:
    df[col] = clean_column(col)

# Check how many missing in normalized-losses
st.subheader("🔍 Normalized Losses: Before and After Cleaning")
st.write(f"Missing values before filling: `{df['normalized-losses'].isna().sum()}`")
st.write(f"Data type: `{df['normalized-losses'].dtype}`")

# Fill missing values in normalized-losses with mean
mean_losses = df['normalized-losses'].mean()
df['normalized-losses'].fillna(mean_losses, inplace=True)
st.write(f"Filled missing values with mean: **{mean_losses:.2f}**")
st.dataframe(df[['normalized-losses']].head(10))  # show a few rows for confirmation

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

print("Part 2")

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Streamlit page config
st.set_page_config(page_title="Automobile Data Analysis", layout="centered")

# Title
st.title("🚗 Part 2: Statistical Analysis of Automobile Dataset")

# Define column names
column_names = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", 
    "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", 
    "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", 
    "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", 
    "price"
]

# Load dataset
df = pd.read_csv("CleanedAutomobile.csv", header=None, names=column_names)

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Convert columns to numeric
cols_to_convert = ['normalized-losses', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'price', 'peak-rpm']
df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

# Show initial preview
st.subheader("📋 Cleaned Data Preview")
st.write("Make sure 'normalized-losses' is now numeric and no '?' remains:")
st.dataframe(df[cols_to_convert].head())

st.markdown("---")

# Q1: Data type of 'peak-rpm'
st.subheader("🔍 Q1: Data Type of 'peak-rpm'")
st.write(f"Data type of **'peak-rpm'**: `{df['peak-rpm'].dtype}`")

st.markdown("---")

# Q2: Correlation Matrix
st.subheader("📊 Q2: Correlation Matrix for Selected Variables")
corr_df = df[['bore', 'stroke', 'compression-ratio', 'horsepower']].dropna()
correlation_matrix = corr_df.corr()
st.write("Correlation Matrix:")
st.dataframe(correlation_matrix.style.background_gradient(cmap='coolwarm').format("{:.2f}"))

# Heatmap
fig1, ax1 = plt.subplots(figsize=(7, 5))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax1)
st.pyplot(fig1)

st.markdown("---")

# Q3a: Correlation between stroke and price
st.subheader("📈 Q3a: Correlation Between 'Stroke' and 'Price'")
corr_data = df[['stroke', 'price']].dropna()
corr_value = corr_data.corr().iloc[0, 1]
st.write(f"The correlation coefficient is: `{corr_value:.3f}`")

st.markdown("---")

# Q3b: Stroke vs Price (with regplot)
st.subheader("📉 Q3b: Linear Relationship Between 'Stroke' and 'Price'")
fig2, ax2 = plt.subplots(figsize=(7, 5))
sns.regplot(x='stroke', y='price', data=corr_data, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
ax2.set_title("Stroke vs Price")
st.pyplot(fig2)

st.write("""
The regression line shows a **weak or no linear relationship**, as the correlation coefficient is close to zero.
""")

st.markdown("---")

# Q4: Group by body-style
st.subheader("💰 Q4: Average Price by Body Style")
price_by_body = df[['body-style', 'price']].dropna()
grouped_price = price_by_body.groupby("body-style")['price'].mean().reset_index().sort_values(by='price', ascending=False)
st.dataframe(grouped_price.style.format({"price": "{:.2f}"}))

# Bar chart
fig3 = px.bar(grouped_price, x='body-style', y='price',
              title="Average Price by Body Style",
              labels={'price': 'Average Price', 'body-style': 'Body Style'},
              color='price', color_continuous_scale='Blues')
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align: center;'>Made by Poornima</div>", unsafe_allow_html=True)


