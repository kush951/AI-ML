# clustering_task.py

from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs

# 1. Generate Synthetic Data
# (Simulating Annual Income & Spending Score)
X, _ = make_blobs(n_samples=300, centers=5, cluster_std=0.60, random_state=0)

# Optional: Convert to DataFrame (better visualization/debugging)
df = pd.DataFrame(X, columns=['Annual Income (k$)', 'Spending Score (1-100)'])

# 2. Find the Optimal K using Elbow Method
wcss = []  # Within-Cluster Sum of Squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# 3. Plot the Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()