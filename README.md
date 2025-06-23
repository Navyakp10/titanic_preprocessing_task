# Titanic Preprocessing Task

This project demonstrates how to clean and prepare data using the Titanic dataset.

## 🔧 Tools Used:
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn

## 🧼 Data Cleaning Steps:
- Handled missing values in Age and Embarked using median and mode
- Dropped the Cabin column due to excessive null values
- Converted categorical features using One-Hot Encoding
- Standardized numerical features (Age, Fare) using StandardScaler
- Removed outliers from the Fare column using the IQR method

## 📁 Files in this Repo:
- titanic.csv – original dataset
- titanic_preprocessing.py – Python script for data cleaning
- titanic_cleaned.csv – cleaned output dataset

## ✅ Output:
Cleaned data ready for machine learning.
