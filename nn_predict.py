import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the trained model
model = tf.keras.models.load_model('my_model.h5')

# Load the saved scaler from the pickle file
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)  

# New inputs to make the prediction
nuevas_entradas = np.array([[5, 20000]])  

# Ensure the scaler is a StandardScaler 
if isinstance(scaler, MinMaxScaler):
    # Scale the new inputs using the loaded scaler
    nuevas_entradas_scaled = scaler.transform(nuevas_entradas)  # Use transform on the loaded scaler
else:
    raise ValueError("Loaded object is not a StandardScaler!")

# Perform the prediction
predictions = model.predict(nuevas_entradas_scaled)

# Display the predictions
print("Predictions:", predictions)

w1, w2, w3, a, b, owa_utility, artm_utility = predictions[0]

# Normalize the weights to sum to 10
weights_sum = w1 + w2 + w3
if weights_sum != 0:  # Avoid division by zero
    w1 = (w1 / weights_sum) * 10
    w2 = (w2 / weights_sum) * 10
    w3 = (w3 / weights_sum) * 10
else:
    print("Warning: The sum of the weights is zero, unable to normalize.")

# Normalize a and b to sum to 10
ab_sum = a + b
if ab_sum != 0:  # Avoid division by zero
    a = (a / ab_sum) * 10
    b = (b / ab_sum) * 10
else:
    print("Warning: The sum of a and b is zero, unable to normalize.")

# Compare ARTM and OWA
if artm_utility > owa_utility:
    # If ARTM is greater, display a and b
    resultado = (a, b)
    print("Result: ARTM with values", resultado)
else:
    # If OWA is greater, display w1, w2, and w3
    resultado = (w1, w2, w3)
    print("Result: OWA with normalized weights", resultado)
