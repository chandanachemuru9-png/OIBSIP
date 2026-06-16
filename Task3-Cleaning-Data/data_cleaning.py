# Task 3 - Data Cleaning
# OIBSIP Internship | Chandana Chemuru

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('messy_data.csv')
print("Original Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 2. Check Missing Values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# 3. Fill Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 4. Remove Duplicates
print("\nDuplicates Before:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicates After:", df.duplicated().sum())
print("Shape After Cleaning:", df.shape)

# 5. Outlier Detection
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.boxplot(df['Salary'], color='lightblue')
plt.title('Salary Outliers')
plt.subplot(1,2,2)
sns.boxplot(df['Purchase_Amount'], color='lightgreen')
plt.title('Purchase Amount Outliers')
plt.tight_layout()
plt.savefig('outliers_boxplot.png')
plt.show()

# 6. Remove Outliers
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Salary'] >= Q1-1.5*IQR) & (df['Salary'] <= Q3+1.5*IQR)]
print("\nShape After Outlier Removal:", df.shape)

# 7. Save Cleaned Data
df.to_csv('cleaned_data.csv', index=False)

# 8. Visualization
plt.figure(figsize=(8,5))
sns.histplot(df['Salary'], bins=20, kde=True, color='blue')
plt.title('Salary Distribution After Cleaning')
plt.tight_layout()
plt.savefig('salary_distribution.png')
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20, kde=True, color='orange')
plt.title('Age Distribution After Cleaning')
plt.tight_layout()
plt.savefig('age_distribution.png')
plt.show()

print("\nData Cleaning Complete!")