import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import streamlit as st

#titanic dataset
data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

#pré-processamento
data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']].dropna()
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

X = data.drop('Survived', axis=1).values
y = data['Survived'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#tensores
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

#define neural network class
class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.fc1 = nn.Linear(6, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

#treinando o modelo
model = TitanicModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

#streamlit 
st.title("Previsão de Sobrevivência no Titanic")
st.write("Preencha os dados do passageiro para prever a sobrevivência")

pclass = st.selectbox("Classe", [1, 2, 3])
sex = st.selectbox("Sexo", ["male", "female"])
age = st.slider("Idade", 1, 80, 25)
sibsp = st.slider("Número de irmãos/cônjuges a bordo", 0, 5, 0)
parch = st.slider("Número de pais/filhos a bordo", 0, 5, 0)
fare = st.slider("Preço da passagem", 0.0, 500.0, 50.0)

sex_val = 0 if sex == "male" else 1
user_input = scaler.transform([[pclass, sex_val, age, sibsp, parch, fare]])
user_tensor = torch.tensor(user_input, dtype=torch.float32)

#predict
with torch.no_grad():
    prediction = model(user_tensor)
    prediction_value = prediction.item()
    survived = prediction_value >= 0.5

st.subheader("Resultado da Previsão")
st.write(f"Probabilidade de sobrevivência: {prediction_value:.2f}")
st.success("Sobreviveu!" if survived else "Não sobreviveu.")