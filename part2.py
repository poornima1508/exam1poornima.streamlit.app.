import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load dataset with column names

column_names = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", 
    "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", 
    "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", 
    "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", 
    "price"
]
df = pd.read_csv('CleanedAutomobile.csv', header=None, names=column_names)


#Question 1: What is the data type of the column "peak-rpm"?

print(df['peak-rpm'].dtype)

#Question 2: Find the correlation between the following columns: bore, stroke, compression-ratio, and horsepower.

df.replace('?', np.nan, inplace=True)

# Convert the relevant columns to numeric, just in case
cols = ['bore', 'stroke', 'compression-ratio', 'horsepower']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

# Select the desired columns
selected_df = df[cols]

# Compute and display the correlation matrix
correlation_matrix = selected_df.corr()
print(correlation_matrix)

#Question 3 a):Find the correlation between x="stroke" and y="price".

# Replace non-numeric placeholders with NaN
df.replace('?', np.nan, inplace=True)

# Convert the relevant columns to numeric
df[['stroke', 'price']] = df[['stroke', 'price']].apply(pd.to_numeric, errors='coerce')

# Calculate the correlation between 'stroke' and 'price'
correlation = df[['stroke', 'price']].corr()

# Display the result
print(correlation)

# Question 3 b) Given the correlation results between "price" and "stroke", do you expect a linear relationship?

# Clean data: replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert to numeric
df[['stroke', 'price']] = df[['stroke', 'price']].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values in either column
df_clean = df[['stroke', 'price']].dropna()

# Plot using regplot
sns.regplot(x='stroke', y='price', data=df_clean)

# Display plot
plt.title("Relationship between Stroke and Price")
plt.xlabel("Stroke")
plt.ylabel("Price")
plt.show()

#Question 4: Use the "groupby" function to find the average "price" of each car based on "body-style".

# Write your code below and press Shift+Enter to execute
average_price_by_body_style = df.groupby('body-style')['price'].mean().reset_index()

# Display the result
print(average_price_by_body_style)

