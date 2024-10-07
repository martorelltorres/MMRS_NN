import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Crear datos simulados con 100 muestras para cada característica
# Número de vehículos exploradores, densidad de objetos de interés, superficie del área de exploración, peso de la información de los objetos
numero_vehiculos = np.random.randint(5, 20, size=(100,))  # 100 muestras de número de vehículos
densidad_objetos = np.random.uniform(0.3, 1.0, size=(100,))  # 100 muestras de densidad de objetos
superficie_area = np.random.uniform(50, 300, size=(100,))  # 100 muestras de superficie del área
peso_informacion = np.random.uniform(0.5, 1.0, size=(100,))  # 100 muestras de peso de la información

# Combinar estas variables en una matriz de características (X) como entrada para el modelo
X = np.column_stack((numero_vehiculos, densidad_objetos, superficie_area, peso_informacion))

# Crear etiquetas de salida simuladas (y), con 100 muestras y 7 parámetros a predecir
# Los 7 parámetros son: a, b, w1, w2, w3, ARTM, OWA
y = np.random.rand(100, 7)  # Simulación de etiquetas para 100 muestras y 7 salidas

# Asegúrate de que los valores de a, b, w1, w2, w3 están en el rango de [0, 1]
y[:, :7] = np.clip(y[:, :7], 0, 1)  

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos (opcional pero recomendado)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir el modelo
model = Sequential()

# Capa de entrada y una capa oculta con 64 neuronas
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))

# Otra capa oculta
model.add(Dense(32, activation='relu'))

# Capa de salida con tantas neuronas como parámetros que deseas predecir
model.add(Dense(y_train.shape[1], activation='linear'))  # 'linear' para regresión

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Resumen del modelo
model.summary()

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)

# Evaluar el rendimiento en los datos de prueba
test_loss, test_mae = model.evaluate(X_test, y_test)
# print(f"Mean Absolute Error on test data: {test_mae}")

# Predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Comparar las predicciones con las etiquetas reales (opcional)
# print("Predicciones:", y_pred)
# print("Valores reales:", y_test)

# Guardar el modelo
model.save('mi_modelo.h5')
