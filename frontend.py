import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_california_housing
from logging import StreamHandler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error

# load dataset
california_housing = fetch_california_housing()
california_df = pd.DataFrame(california_housing.data,columns = california_housing.feature_names)
california_df["Price"] = california_housing.target


# Title of the app
st.title("california housing price predictiion")

# Data Overview
st.header("data overview for first 10 row")
st.write(california_df.head(10))

# split the data into input and output
X = california_df.drop('Price', axis=1) # input features
y = california_df['Price'] # target
X_train, X_test,y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0)

# standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Selection
sel_mod = st.selectbox("select a model", ["Linear Reg ression", "Random Forest", "Decision Tree"])

models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(),
          "Decision Tree": DecisionTreeRegressor()}

# Train the model
selected = models[sel_mod] # initialises the model

# train the selected model
selected.fit(X_train, y_train)

# make predictions
y_pred = selected.predict(X_test)

# model evaluation
R2 = r2_score(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)

# display the units
st.write(f"R2score: {R2}")
st.write(f"Mean_Squared_Error: {MSE}")
st.write(f"mean_Absolute_Error: {MAE}")

# prompt for user input
st.write("enter the input values for prediction:")

user_input = {}
for column in X.columns:
    user_input[column] = st.number_input(column, min_value = np.min(X[column]), max_value = np.max(X[column]), value = np.mean(X[column]))
    
# convert dictionary to dataframe
user_input_df = pd.DataFrame([user_input])

# standardise the dataframe
user_input_sc_df = scaler.transform(user_input_df)

# make predictions for the price
price_predicted = selected.predict(user_input_sc_df)

# display the predicted price
st.write(f"predicted price is: {price_predicted[0] * 100000}")