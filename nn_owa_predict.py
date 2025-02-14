import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = tf.keras.models.load_model('owa_model.keras')

# Load the saved scaler from the pickle file
with open('owa_scaler.pkl', 'rb') as f:
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

# Expecting the model to output 4 values: [w1, w2, w3, utility]
if predictions.shape[1] != 4:
    raise ValueError("Expected the model to output 4 values (w1, w2, w3, utility).")

# Unpack the predictions
w1, w2, w3, utility = predictions[0]

# Normalize w1, w2, w3 so that they sum to 10
total = w1 + w2 + w3
if total > 0:
    w1 = (w1 / total) * 10
    w2 = (w2 / total) * 10
    w3 = (w3 / total) * 10
else:
    w1, w2, w3 = 10/3, 10/3, 10/3  # Assign equal weights if the total is zero

print("w1 =", w1)
print("w2 =", w2)
print("w3 =", w3)
print("Utility =", utility)

