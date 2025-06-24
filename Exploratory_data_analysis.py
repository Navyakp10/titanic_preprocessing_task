import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the Titanic dataset
df = pd.read_csv('Titanic-Dataset.csv')

# Display first few rows
print("First 5 rows:")
print(df.head())

# Dataset shape and types
print("\nDataset Info:")
print(df.info())

# Summary statistics for numeric columns
print("\nSummary Statistics:")
print(df.describe())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Histograms for numeric columns
df.hist(bins=30, figsize=(15, 10), color='skyblue')
plt.suptitle("Histograms of Numeric Features")
plt.show()

# Boxplots for numeric columns
numeric_cols = df.select_dtypes(include='number').columns

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f'Boxplot of {col}')
    plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# Pairplot of selected columns (to avoid overload)
selected = ['Survived', 'Pclass', 'Age', 'Fare']
sns.pairplot(df[selected], hue='Survived')
plt.suptitle("Pairplot", y=1.02)
plt.show()

# Optional: Plotly interactive scatter plot
fig = px.scatter(df, x='Age', y='Fare', color='Survived', title='Age vs Fare by Survival')
fig.show()
