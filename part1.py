import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Title of the app
st.title('Part 1: Data Exploration and Transformations')

# Introduction text
st.write("""
Welcome to the data exploration app. In this app, we perform multiple data transformations on an automobile dataset:
- Replace NaN values in the 'stroke' column with the mean value
- Convert the 'highway-mpg' column to 'highway-L/100km'
- Normalize the 'height' column
- Create indicator variables for 'aspiration'
""")

# Load the dataset
df = pd.read_csv('auto.csv', header=None)
column_names = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]  
df = pd.read_csv('auto.csv', header=None, names=column_names)

# Display the first few rows of the data
st.subheader("Dataset Overview")
st.write("Here are the first few rows of the dataset:")
st.write(df.head())

# Display the column names
st.write("Columns in the dataset:")
st.write(df.columns.tolist())

# Replace NaN in "stroke" column with the mean value
st.subheader("Question 1: Replace NaN in 'stroke' Column")
avg_stroke = pd.to_numeric(df['stroke'], errors='coerce').mean()
df['stroke'] = df['stroke'].replace('?', np.nan).astype('float')
df["stroke"].fillna(avg_stroke, inplace=True)
st.write(f"Replaced NaN values in 'stroke' with the mean: {avg_stroke:.2f}")

# Convert highway-mpg to L/100km and change column name
st.subheader("Question 2: Convert 'highway-mpg' to 'highway-L/100km'")
df["highway-L/100km"] = 235 / df["highway-mpg"]
st.write("Transformed 'highway-mpg' to 'highway-L/100km'. Here's the updated data:")
st.write(df[['highway-mpg', 'highway-L/100km']].head())

# Normalize the 'height' column
st.subheader("Question 3: Normalize the 'height' Column")
df['height'] = df['height'] / df['height'].max()
st.write("Normalized the 'height' column. Here's the updated data:")
st.write(df[['height']].head())

# Create indicator variable for 'aspiration'
st.subheader("Question 4: Create Indicator Variable for 'Aspiration'")

# Check unique values in 'aspiration' to ensure they are correct
st.write("Unique values in 'aspiration' column:", df['aspiration'].unique())

# Create dummy variables for 'aspiration'
dummy_variable_2 = pd.get_dummies(df["aspiration"], prefix='aspiration')

# Check the created columns
st.write("Columns after dummy variable creation:", dummy_variable_2.columns.tolist())

df = pd.concat([df, dummy_variable_2], axis=1)
df.drop("aspiration", axis=1, inplace=True)

# Display the new dataframe with the dummy variables
st.write("New columns created after dummy variable transformation:")
st.write(df[['aspiration-std', 'aspiration-turbo']].head())


# Show final dataframe shape
st.subheader("Final Dataset Information")
st.write(f"Number of rows: {df.shape[0]}")
st.write(f"Number of columns: {df.shape[1]}")

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

