import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Carregando o dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

#Selecionando features simples para exemplo
df = df[["Survived", "Pclass", "Age", "Sex"]].dropna()
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

X = df[["Pclass", "Age", "Sex"]]
y = df["Survived"]

#Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Modelo
model = LogisticRegression()
model.fit(X_train, y_train)

#Previsão
y_pred = model.predict(X_test)

#Avaliação
print("Acurácia:", accuracy_score(y_test, y_pred))