import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#dados sintéticos
X = 2 * np.random.rand(100, 1)  #100 valores aleatórios entre 0 e 2
y = 4 + 3 * X + np.random.randn(100, 1)  #y = 4 + 3x + ruído

#dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Criando o modelo de regressão linear
model = LinearRegression()

#Treinando o modelo com os dados de treino
model.fit(X_train, y_train)

#Fazendo previsões com os dados de teste
y_pred = model.predict(X_test)

#Avaliando o erro (quanto o modelo erra na média)
mse = mean_squared_error(y_test, y_pred)

print("Coeficiente angular (inclinação):", model.coef_)   # Esperado≈ 3
print("Intercepto:", model.intercept_)                    # Esperado≈ 4
print("Erro médio quadrático:", mse)

#Visualização
plt.scatter(X_test, y_test, color="blue", label="Real")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Previsão")
plt.title("Regressão Linear")
plt.legend()
plt.show()