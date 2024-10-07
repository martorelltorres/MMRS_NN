import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Cargar el modelo entrenado
model = tf.keras.models.load_model('mi_modelo.h5')

# Nuevas entradas (ejemplo)
# Definir las nuevas entradas (número de vehículos, densidad de objetos, superficie del área, peso de la información)
nuevas_entradas = np.array([[12, 0.6, 110, 0.9]])  # Ejemplo con una nueva muestra

# Escalar las nuevas entradas utilizando el mismo escalador que usaste para los datos de entrenamiento
scaler = StandardScaler()
# Escalar las nuevas entradas
nuevas_entradas_scaled = scaler.fit_transform(nuevas_entradas)

# Realizar la predicción
predicciones = model.predict(nuevas_entradas_scaled)

# Mostrar las predicciones
print("Predicciones:", predicciones)

# Extraer los valores
a, b, w1, w2, w3, ARTM, OWA = predicciones[0]

# Comparar ART y OWA
if ARTM > OWA:
    # Si ART es mayor, mostrar a y b
    resultado = (a, b)
    # Mostrar el resultado
    print("Resultado: ARTM con pesos", resultado)
else:
    # Si OWA es mayor, mostrar w1, w2, y w3
    resultado = (w1, w2, w3)
    print("Resultado: OWA con pesos", resultado)

