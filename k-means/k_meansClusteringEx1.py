from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# criando dados fictícios agrupáveis
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# modelo KMeans com 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# revendo os clusters
y_kmeans = kmeans.predict(X)

# visualizando
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')
plt.title("Agrupamento com K-Means")
plt.show()