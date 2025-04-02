import numpy as np
import tensorflow as tf
import pickle
from scipy.optimize import differential_evolution

# Load the trained model
model = tf.keras.models.load_model('owa_model.keras', compile=False)

# Load the scalers
with open('owa_scalers.pkl', 'rb') as f:
    scaler_X, scaler_y = pickle.load(f)

# Function to predict utility
def predict_utility(auv_count, area, w1, w2, w3):
    input_data = np.array([[auv_count, area, w1, w2, w3]])
    input_scaled = scaler_X.transform(input_data)
    utility_scaled = model.predict(input_scaled)
    utility = scaler_y.inverse_transform(utility_scaled.reshape(-1, 1))[0, 0]
    return utility

# Objective function to maximize utility with penalty for constraint violation
def objective(w, auv_count, area):
    w1, w2, w3 = w
    return -predict_utility(auv_count, area, w1, w2, w3)  # Minimizar negativo 

# Function to find optimal w1, w2, w3
def find_optimal_w(auv_count, area):
    bounds = [(0, 10), (0, 10), (0, 10)]
    result = differential_evolution(objective, bounds, args=(auv_count, area), strategy='best1bin', maxiter=1000, popsize=15)
    optimal_w1, optimal_w2, optimal_w3 = result.x
    max_utility = -result.fun  # Since we minimized negative utility
    return optimal_w1, optimal_w2, optimal_w3, max_utility

# Example usage
if __name__ == "__main__":
    # Get user input for auv_count and area
    auv_count = int(input("Number of explorrer AUVs: "))
    area = int(input("Area exploration suurface: "))
    w1, w2, w3, utility = find_optimal_w(auv_count, area)
    print(f"Optimal w1: {w1}, w2: {w2}, w3: {w3}, Utility: {utility}")
