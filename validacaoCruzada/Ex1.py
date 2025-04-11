from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

#dataset
data = fetch_california_housing()
X, y = data.data, data.target

#Modelo
model = LinearRegression()

#Cross-validation com 5 divisões
scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)

print("Erros quadráticos médios (negativos):", scores)
print("Erro médio quadrático (positivo):", -scores.mean())