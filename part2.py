import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Configure Streamlit page
st.set_page_config(page_title="Automobile Data Part 2", layout="centered")

# Title
st.title("🚗 Part 2: Statistical Analysis of Automobile Dataset")

# Load dataset
column_names = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", 
    "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", 
    "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", 
    "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", 
    "price"
]
df = pd.read_csv("CleanedAutomobile.csv", header=None, names=column_names)

# Replace '?' with NaN and convert all relevant columns to numeric
cols_to_convert = ['bore', 'stroke', 'compression-ratio', 'horsepower', 'price', 'peak-rpm']
df.replace("?", np.nan, inplace=True)
df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

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
The regression line shows a very **weak or no linear relationship**, as the correlation coefficient is close to zero.
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
st.markdown("<div style='text-align: center;'>Made with ❤️ by Poornima</div>", unsafe_allow_html=True)
