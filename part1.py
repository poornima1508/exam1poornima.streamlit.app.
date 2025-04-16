import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Title of the app
st.title('Part 1: Automobile Data Exploration and Transformations')

# Introduction text
st.write("""
This app explores a dataset of automobiles. We will perform the following data transformations and visualizations:
1. Handle missing values in the 'stroke' column.
2. Convert 'highway-mpg' to 'highway-L/100km'.
3. Normalize the 'height' column.
4. Create dummy variables for the 'aspiration' column.
5. Visualize key relationships in the data.
""")

# Load the dataset
df = pd.read_csv('auto.csv', header=None)
column_names = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", 
    "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", 
    "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", 
    "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", 
    "price"
]
df = pd.read_csv('auto.csv', header=None, names=column_names)

# Dataset Overview
st.subheader("Dataset Overview")
st.write("Here are the first few rows of the dataset:")
st.write(df.head())

# Display the column names
st.write("Columns in the dataset:", df.columns.tolist())

# ==============================================
# Question 1: Handle missing values in 'stroke' column
st.subheader("Question 1: Handle Missing Values in 'Stroke' Column")
avg_stroke = pd.to_numeric(df['stroke'], errors='coerce').mean()  # Calculate average stroke, ignoring non-numeric
df['stroke'] = df['stroke'].replace('?', np.nan).astype('float')  # Replace '?' with NaN
df["stroke"].fillna(avg_stroke, inplace=True)  # Replace NaN with the average stroke
st.write(f"Replaced NaN values in 'stroke' with the mean: {avg_stroke:.2f}")

# ==============================================
# Question 2: Convert 'highway-mpg' to 'highway-L/100km'
st.subheader("Question 2: Convert 'highway-mpg' to 'highway-L/100km'")
df["highway-L/100km"] = 235 / df["highway-mpg"]  # Convert to L/100km
st.write("Transformed 'highway-mpg' to 'highway-L/100km'. Here's the updated data:")
st.write(df[['highway-mpg', 'highway-L/100km']].head())

# ==============================================
# Question 3: Normalize the 'height' column
st.subheader("Question 3: Normalize the 'Height' Column")
df['height'] = df['height'] / df['height'].max()  # Normalize the height column
st.write("Normalized the 'height' column. Here's the updated data:")
st.write(df[['height']].head())

# ==============================================
# Question 4: Create Dummy Variables for 'aspiration'
st.subheader("Question 4: Create Dummy Variables for 'Aspiration'")

# Check unique values in 'aspiration' to ensure they are correct
st.write("Unique values in 'aspiration' column:", df['aspiration'].unique())

# Create dummy variables for 'aspiration'
dummy_variable_2 = pd.get_dummies(df["aspiration"], prefix='aspiration')
df = pd.concat([df, dummy_variable_2], axis=1)  # Merge with original DataFrame
df.drop("aspiration", axis=1, inplace=True)  # Drop original 'aspiration' column

# Display the new dataframe with the dummy variables
st.write("New columns created after dummy variable transformation:")
st.write(df.head())

# ==============================================
# Final Dataset Information
st.subheader("Final Dataset Information")
st.write(f"Number of rows: {df.shape[0]}")
st.write(f"Number of columns: {df.shape[1]}")

# ==============================================
# Visualizations

# Histogram of 'height' column
st.subheader("Histogram of 'Height' Column (Normalized)")
fig = plt.figure(figsize=(8, 6))
sns.histplot(df['height'], bins=30, kde=True)
st.pyplot(fig)

# Scatter plot of 'horsepower' vs 'curb-weight'
st.subheader("Scatter Plot: 'Horsepower' vs 'Curb-weight'")
fig2 = px.scatter(df, x='horsepower', y='curb-weight', title="Horsepower vs Curb-weight")
st.plotly_chart(fig2)

# Box plot for 'price' by 'body-style'
st.subheader("Box Plot: 'Price' by 'Body Style'")
fig3 = plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='body-style', y='price')
st.pyplot(fig3)

# Line plot showing the trend of 'price' over 'engine-size'
st.subheader("Line Plot: 'Price' vs 'Engine Size'")
fig4 = px.line(df, x='engine-size', y='price', title="Price vs Engine Size")
st.plotly_chart(fig4)
