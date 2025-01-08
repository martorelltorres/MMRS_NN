#!/usr/bin/env python3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Import EarlyStopping
from tensorflow.keras.regularizers import l2  # Import L2 regularizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, SGD

df = pd.read_csv('/home/uib/MRS_data/NN/consolidated_data.csv')  

X = df[['auv_count', 'area']].values  

y = df[['w1', 'w2', 'w3', 'a', 'b', 'owa_utility','artm_utility']].values  

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)

# Escalar los datos (opcional pero recomendado)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

X_test = scaler.transform(X_test)

# Definir el modelo
model = Sequential()

# Capa de entrada y una capa oculta con 128 neuronas y regularización L2
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)))  # L2 regularization
# model.add(Dropout(0.1))  # Add Dropout layer (20% dropout)

# Otra capa oculta con 32 neuronas y regularización L2
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))  # L2 regularization
# model.add(Dropout(0.1))  # Add Dropout layer (20% dropout)

# Capa de salida con tantos neuronas como parámetros se quieren predecir, con regularización L2
model.add(Dense(y_train.shape[1], activation='linear', kernel_regularizer=l2(0.01)))  # L2 regularization

# OPTIMIZERS
optimizer = Adam(learning_rate=0.001) 
model.compile(optimizer=optimizer, 
              loss='mean_squared_error', 
              metrics=['mean_absolute_error', 'mean_squared_error'])

# optimizer = Nadam(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

# optimizer = RMSprop(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

# optimizer = SGD(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])


# Resumen del modelo
model.summary()

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=20,         # Number of epochs without improvement before stopping
    restore_best_weights=True,  # Restore the best weights when stopping
    verbose=1            # Print stopping messages
)

# Define the ReduceLROnPlateau callback
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,        # Reduce learning rate by a factor of 0.5
    patience=5,        # Wait for 5 epochs without improvement before reducing
    min_lr=1e-6,       # Minimum learning rate to avoid going too small
    verbose=1          # Print learning rate adjustments
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stopping, lr_scheduler]
)

# Evaluate the model on the test set
test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=1)

# Predictions
y_pred = model.predict(X_test)

# Guardar el modelo
model.save('my_model.h5')

# Get the training and validation loss and metrics
history_dict = history.history

# Plot training and validation loss
plt.plot(history_dict['loss'], label='Training Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Plot training and validation mean absolute error
plt.plot(history_dict['mean_absolute_error'], label='Training MAE')
plt.plot(history_dict['val_mean_absolute_error'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.title('Training and Validation MAE')
plt.show()

# Plot training and validation mean absolute error
plt.plot(history_dict['mean_squared_error'], label='Training MSE')
plt.plot(history_dict['val_mean_squared_error'], label='Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Training and Validation Mean Squared Error')
plt.show()
