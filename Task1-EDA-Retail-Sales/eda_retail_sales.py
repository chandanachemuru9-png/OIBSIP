import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('retail_sales.csv')
print("Shape:", df.shape)
print(df.head())
print(df.describe())

plt.figure(figsize=(8,5))
df.groupby('Product_Category')['Total_Amount'].sum().sort_values().plot(kind='bar', color='steelblue')
plt.title('Total Sales by Product Category')
plt.tight_layout()
plt.savefig('sales_by_category.png')
plt.show()

plt.figure(figsize=(6,5))
df.groupby('Gender')['Total_Amount'].sum().plot(kind='pie', autopct='%1.1f%%')
plt.title('Sales by Gender')
plt.tight_layout()
plt.savefig('sales_by_gender.png')
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20, kde=True, color='purple')
plt.title('Age Distribution')
plt.tight_layout()
plt.savefig('age_distribution.png')
plt.show()

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M')
df.groupby('Month')['Total_Amount'].sum().plot(kind='line', marker='o', color='green', figsize=(12,5))
plt.title('Monthly Sales Trend')
plt.tight_layout()
plt.savefig('monthly_trend.png')
plt.show()

plt.figure(figsize=(7,5))
sns.heatmap(df[['Age','Quantity','Price_per_Unit','Total_Amount']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

print("EDA Complete!")