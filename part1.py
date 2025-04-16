import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Title of the app
st.title('Part 1')

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/auto.csv')


#Question #1: replace NaN in "stroke" column with the mean value.
st.subheader("Column names in the dataset")
st.write(df.columns.tolist())

st.subheader("First few rows of data")
st.write(df.head())

avg_stroke=df['stroke'].astype('float').mean(axis=0)
print("Average of stroke:", avg_stroke)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)

#Question #2: transform mpg to L/100km in the column of "highway-mpg" and change the name of column to "highway-L/100km".

# Convert highway-mpg to L/100km
df["highway-L/100km"] = 235 / df["highway-mpg"]

# Check the transformed data
df.head()

#Question #3: normalize the column "height".

df['height'] = df['height'] / df['height'].max()
df.head()

#Question #4: Similar to before, create an indicator variable for the column "aspiration"

# Create dummy variables for 'aspiration'
dummy_variable_2 = pd.get_dummies(df["aspiration"])

# Rename columns for clarity
dummy_variable_2.rename(columns={'std': 'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)

# Merge with original DataFrame
df = pd.concat([df, dummy_variable_2], axis=1)

# Drop the original 'aspiration' column
df.drop("aspiration", axis=1, inplace=True)

# Optional: view the updated DataFrame
df.head()

#Question #5: Merge the new dataframe to the original (Dataframe after Question 4) dataframe, then drop the column 'aspiration'.

# Merge with the existing DataFrame
df = pd.concat([df, dummy_variable_2], axis=1)
df.head()

#Question #6: How many Rows and Columns do you have in the final Dataframe?

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
