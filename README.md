Singapore Resale Price Prediction
This project aims to predict the resale prices of Housing and Development Board (HDB) flats in Singapore using machine learning techniques. The project involves end-to-end processes, from data collection and preprocessing to model development, evaluation, and deployment in a user-friendly Streamlit web application.

Project Overview

This project encompasses the following key stages:

Python Scripting, Data Preprocessing, EDA, Machine Learning, and Streamlit

Data Collection and Preprocessing:

Data Source: Downloaded historical resale flat data from official HDB sources, covering the period from 1990 to the present.
Initial Cleaning: Handled missing values, corrected inconsistencies, and ensured data integrity.
Feature Engineering: Enhanced the dataset by creating new features and transforming existing ones to better capture the underlying patterns.

Data Exploration and Handling:

Outlier Detection: Identified and handled outliers to ensure the model's robustness.
Categorical Encoding: Encoded categorical features using techniques such as Label Encoding to convert them into numerical formats suitable for machine learning algorithms.

Model Selection and Training:

Cross-Validation: Tested various regression models (e.g., Linear Regression, Random Forest Regressor, etc.).
Model Evaluation: Selected the RandomForest Regressor based on superior performance in terms of R-squared and Mean Squared Error (MSE).
Final Model: The RandomForest Regressor was chosen as the best model for predicting resale prices.

Model Deployment:

Model Serialization: The trained RandomForest Regressor model was saved using joblib for later use in the application.
Interactive Dashboard: Developed an interactive dashboard using Streamlit, allowing users to input relevant features and get predictions on flat resale prices.
