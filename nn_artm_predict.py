import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = tf.keras.models.load_model('artm_model.keras')

# Load the saved scaler from the pickle file
with open('artm_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Get user input for auv_count and area
auv_count = int(input("Number of AUVs: "))
area = int(input("Area exploration: "))

# New inputs for prediction
nuevas_entradas = np.array([[auv_count, area]])

# Ensure the scaler is a StandardScaler and scale the new inputs
if isinstance(scaler, StandardScaler):
    nuevas_entradas_scaled = scaler.transform(nuevas_entradas)
else:
    raise ValueError("Loaded object is not a StandardScaler!")

# Perform the prediction
predictions = model.predict(nuevas_entradas_scaled)

# Expecting the model to output 3 values: [a, b, utility]
if predictions.shape[1] != 3:
    raise ValueError("Expected the model to output 3 values (a, b, utility).")

# Unpack the predictions
a, b, utility = predictions[0]

# Normalize a and b so that they sum to 10
total = a + b
if total > 0:
    a = (a / total) * 10
    b = (b / total) * 10
else:
    a, b = 5, 5  # Assign equal values if the total is zero

print("alpha =", a)
print("beta =", b)
print("Utility =", utility)

