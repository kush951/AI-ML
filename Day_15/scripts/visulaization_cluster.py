# 4. Applying KMeans to the dataset
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from Day_15.scripts.clustering_task import X

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
# 5. Visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow',
label='Centroids')
plt.title('Clusters of Users')
plt.legend()
plt.show()