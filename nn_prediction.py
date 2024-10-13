import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = tf.keras.models.load_model('my_model.h5')

# Load the saved scaler from the pickle file
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)  # Ensure the scaler object is loaded correctly

# New inputs (example)
nuevas_entradas = np.array([[12, 110, 0.9]])  # Example with one new sample

# Ensure the scaler is a StandardScaler 
if isinstance(scaler, StandardScaler):
    # Scale the new inputs using the loaded scaler
    nuevas_entradas_scaled = scaler.transform(nuevas_entradas)  # Use transform on the loaded scaler
else:
    raise ValueError("Loaded object is not a StandardScaler!")

# Perform the prediction
predicciones = model.predict(nuevas_entradas_scaled)

# Display the predictions
print("Predictions:", predicciones)

# Extract the predicted values
a, b, w1, w2, w3, ARTM, OWA = predicciones[0]

# Compare ARTM and OWA
if ARTM > OWA:
    # If ARTM is greater, display a and b
    resultado = (a, b)
    print("Result: ARTM with weights", resultado)
else:
    # If OWA is greater, display w1, w2, and w3
    resultado = (w1, w2, w3)
    print("Result: OWA with weights", resultado)
