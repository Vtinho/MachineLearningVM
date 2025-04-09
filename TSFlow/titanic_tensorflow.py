import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]  #seleciona as colunas úteis
df.dropna(inplace=True)

df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

X = df.drop('Survived', axis=1)
y = df['Survived']

scaler = StandardScaler()
X = scaler.fit_transform(X)

#divide treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#modelo com TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

#compila
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Treinar
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

#avalia
loss, acc = model.evaluate(X_test, y_test)
print(f"Acurácia final: {acc:.2f}")
