#!/usr/bin/env python3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

# Load training data
train_df = pd.read_csv('/home/uib/MMRS_NN/data/may/owa_data_train.csv')  
input_train = train_df[['auv_count', 'area', 'w1', 'w2', 'w3']].values  
output_train = train_df[['utility']].values  

# Load evaluation data
test_df = pd.read_csv('/home/uib/MMRS_NN/data/may/owa_data_test.csv')  
input_test = test_df[['auv_count', 'area', 'w1', 'w2', 'w3']].values  
output_test = test_df[['utility']].values  

# Convert data types for TensorFlow compatibility
input_train = input_train.astype(np.float32)
output_train = output_train.astype(np.float32)
input_test = input_test.astype(np.float32)
output_test = output_test.astype(np.float32)

# Scale the input features
scaler = StandardScaler()
input_train = scaler.fit_transform(input_train)
input_test = scaler.transform(input_test)

# Save the scaler for future use
with open('owa_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_train.shape[1],), kernel_regularizer=l2(0.01)),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(output_train.shape[1], activation='linear', kernel_regularizer=l2(0.01))
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])

# Model summary
model.summary()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Train the model
history = model.fit(
    input_train, output_train,
    epochs=500,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stopping, lr_scheduler]
)

# Evaluate the model on the test set
test_loss, test_mae, test_mse = model.evaluate(input_test, output_test, verbose=1)

# Save the trained model
model.save('owa_model.keras')

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Plot training and validation MAE
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.title('Training and Validation MAE')
plt.show()

# Plot training and validation MSE
plt.plot(history.history['mean_squared_error'], label='Training MSE')
plt.plot(history.history['val_mean_squared_error'], label='Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Training and Validation MSE')
plt.show()
