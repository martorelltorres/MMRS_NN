import numpy as np
import tensorflow as tf
import pickle
from scipy.optimize import differential_evolution

# Load the trained model
model = tf.keras.models.load_model('owa_model.keras', compile=False)

# Load both scalers
with open('owa_scaler.pkl', 'rb') as f:
    scaler_X, scaler_y = pickle.load(f)

# Function to predict utility
def predict_utility(auv_count, area, w1, w2, w3):
    input_data = np.array([[auv_count, area, w1, w2, w3]])
    input_scaled = scaler_X.transform(input_data)
    utility_scaled = model.predict(input_scaled)
    utility = scaler_y.inverse_transform(utility_scaled.reshape(-1, 1))[0, 0]
    return utility

def constraint_transform(x, y):
    w1 = x
    w2 = y
    w3 = 10 - (w1 + w2)  # Ensure the sum is 10
    return [w1, w2, w3]

# Objective function to maximize utility 
def constrained_objective(x, auv_count, area):
    w1, w2, w3 = constraint_transform(x[0], x[1])
    if w3 < 0:
        return 1e6 
    
    return -predict_utility(auv_count, area, w1, w2, w3)  

# Function to find optimal w1, w2, w3
def find_optimal_w(auv_count, area):
    bounds = [(0, 10), (0, 10)] 

    result = differential_evolution(
        constrained_objective, bounds, 
        args=(auv_count, area), 
        strategy='best1bin', 
        maxiter=1000, 
        popsize=15
    )

    # Get valid values
    w1, w2, w3 = constraint_transform(result.x[0], result.x[1])
    max_utility = -result.fun  # Since we minimized negative utility

    return w1, w2, w3, max_utility

# Example usage
if __name__ == "__main__":
    # Get user input for auv_count and area
    auv_count = int(input("Number of explorrer AUVs: "))
    area = int(input("Area exploration surface: "))
    w1, w2, w3, utility = find_optimal_w(auv_count, area)
    print(f"Optimal w1: {w1}, w2: {w2}, w3: {w3}, Utility: {utility}")
