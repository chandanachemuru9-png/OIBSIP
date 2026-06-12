# Task 2 - Customer Segmentation Analysis
# OIBSIP Internship | Chandana Chemuru

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load Data
df = pd.read_csv('customers.csv')
print("Shape:", df.shape)
print(df.head())
print(df.describe())

# 2. Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Scale the Features
features = ['Age', 'Annual_Income', 'Spending_Score', 'Purchase_Frequency']
scaler = StandardScaler()
scaled = scaler.fit_transform(df[features])

# 4. Find Best K (Elbow Method)
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), inertia, marker='o', color='blue')
plt.title('Elbow Method - Finding Best K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.tight_layout()
plt.savefig('elbow_curve.png')
plt.show()

# 5. Apply KMeans with K=4
km = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = km.fit_predict(scaled)

# 6. Visualize Clusters
plt.figure(figsize=(8,6))
colors = ['red','blue','green','orange']
for i in range(4):
    cluster = df[df['Cluster']==i]
    plt.scatter(cluster['Annual_Income'], cluster['Spending_Score'],
                label=f'Cluster {i}', s=60)
plt.title('Customer Segments - Income vs Spending')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.tight_layout()
plt.savefig('clusters_scatter.png')
plt.show()

# 7. Cluster Summary
summary = df.groupby('Cluster')[features].mean()
print("\nCluster Summary:")
print(summary)

summary.plot(kind='bar', figsize=(10,6))
plt.title('Average Features per Cluster')
plt.tight_layout()
plt.savefig('cluster_summary.png')
plt.show()

print("\nCustomer Segmentation Complete!")