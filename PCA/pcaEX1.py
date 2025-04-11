from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# carregando o dataset
data = load_iris()
X = data.data
y = data.target

# aplicando PCA para reduzir para 2 dimensões
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# plotando o resultado
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("PCA no dataset Iris")
plt.show()