import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar dados
data = fetch_california_housing()
X = data.data
y = data.target

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir em treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelo com TensorFlow
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Treinar
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Avaliar
loss, mae = model.evaluate(X_test, y_test)
print(f"MAE: {mae:.4f}")