import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the trained model
model = tf.keras.models.load_model('artm_model.keras')

# Load the saved scaler from the pickle file
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)  

# New inputs to make the prediction
nuevas_entradas = np.array([[5, 20000]])  

# Ensure the scaler is a StandardScaler 
if isinstance(scaler, StandardScaler):
    # Scale the new inputs using the loaded scaler
    nuevas_entradas_scaled = scaler.transform(nuevas_entradas)  # Use transform on the loaded scaler
else:
    raise ValueError("Loaded object is not a StandardScaler!")

# Perform the prediction
predictions = model.predict(nuevas_entradas_scaled)

# Display the predictions
print("Predictions:", predictions)

a, b, utility = predictions[0]
result = (a, b)
print("Result: ARTM with values", result)
