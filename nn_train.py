import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Create simulated data with 100 samples for each feature
# Number of exploring vehicles, object density, exploration area size, information weight
numero_vehiculos = np.random.randint(5, 20, size=(100,))  # 100 samples of number of vehicles
superficie_area = np.random.uniform(50, 300, size=(100,))  # 100 samples of exploration area size
peso_informacion = np.random.uniform(0.5, 1.0, size=(100,))  # 100 samples of information weight

# Combine these variables into a feature matrix (X) as input for the model
X = np.column_stack((numero_vehiculos, superficie_area, peso_informacion))

# OUTPUTS
# Define the number of samples
num_samples = 100

# Simulate values for each parameter
a = np.random.rand(num_samples)  # Values of 'a' between 0 and 1
b = np.random.rand(num_samples)  # Values of 'b' between 0 and 1
w1 = np.random.rand(num_samples)  # Values of 'w1' between 0 and 1
w2 = np.random.rand(num_samples)  # Values of 'w2' between 0 and 1
w3 = np.random.rand(num_samples)  # Values of 'w3' between 0 and 1
ARTM = np.random.rand(num_samples)  # Values of 'ARTM' between 0 and 1
OWA = np.random.rand(num_samples)  # Values of 'OWA' between 0 and 1

# Combine all outputs into a single array for training
y = np.column_stack((a, b, w1, w2, w3, ARTM, OWA))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data (optional but recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Scaler saved successfully!")

X_test = scaler.transform(X_test)

# Define the model
model = Sequential()

# Input layer and one hidden layer with 64 neurons
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))

# Another hidden layer
model.add(Dense(32, activation='relu'))

# Output layer with as many neurons as parameters you want to predict
model.add(Dense(y_train.shape[1], activation='linear'))  # 'linear' for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)

# Evaluate performance on test data
test_loss, test_mae = model.evaluate(X_test, y_test)
# print(f"Mean Absolute Error on test data: {test_mae}")

# Predictions on the test set
y_pred = model.predict(X_test)

# Compare predictions with actual labels (optional)
# print("Predictions:", y_pred)
# print("Actual values:", y_test)

# Save the model
model.save('my_model.h5')
