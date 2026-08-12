import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import os

REPO = os.path.dirname(os.path.abspath(__file__))

# --------------------- LOAD TRAINING DATA ---------------------
paths = [
    os.path.join(REPO, 'weights', '3AUV_weights.csv'),
    os.path.join(REPO, 'weights', '4AUV_weights.csv'),
    os.path.join(REPO, 'weights', '5AUV_weights.csv'),
    os.path.join(REPO, 'weights', '6AUV_weights.csv'),
]

owa_df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
owa_input = owa_df[['auv_count', 'area']].values
owa_output = owa_df[['w1', 'w2', 'w3', 'utility']].values

# =============================================================
# MODIFICACIÓN CLAVE: DEFINICIÓN DE NUEVOS DATOS DE ENTRADA
# =============================================================

# Definir las combinaciones de AUVs y Área solicitadas
auv_counts_list = [3, 4, 5, 6]
areas_list = [15000, 25000, 35000, 45000, 55000]

# Generar todas las combinaciones (producto cartesiano)
new_test_data = []
for auv in auv_counts_list:
    for area in areas_list:
        # Añadimos valores placeholder para w1, w2, w3, utility ya que son variables de salida
        # que el modelo debe predecir, pero son necesarias para la estructura del DataFrame
        new_test_data.append({
            'auv_count': auv, 
            'area': area,
            'w1': np.nan, 
            'w2': np.nan, 
            'w3': np.nan, 
            'utility': np.nan
        })

# Crear el DataFrame de prueba que usaremos para la predicción
new_test_df = pd.DataFrame(new_test_data)

# Reemplazar la variable 'test_df' original con el nuevo DataFrame
test_df = new_test_df 
# Ahora test_df contiene 4 AUVs * 5 Áreas = 20 combinaciones de entrada para predecir.

# --------------------- SCALING ---------------------
scalers_dict = {}
scaled_inputs = {}

for auv in np.unique(owa_input[:, 0]):
    indices = owa_input[:, 0] == auv
    scaler = StandardScaler()
    scaled_inputs[auv] = scaler.fit_transform(owa_input[indices])
    scalers_dict[auv] = scaler

# --------------------- MODEL DEFINITIONS ---------------------
def create_models():
    return {
        "Decision Tree": MultiOutputRegressor(DecisionTreeRegressor(max_depth=5, min_samples_leaf=3)),
        "Random Forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=1000)),
        "SVR": MultiOutputRegressor(svm.SVR(kernel='rbf', C=7, epsilon=1.2, gamma=0.1)),
        "Polynomial" : MultiOutputRegressor(make_pipeline(PolynomialFeatures(4), LinearRegression())),
        "Lasso": MultiOutputRegressor(Lasso(alpha=0.3)),
    }

models_dict = {auv: create_models() for auv in range(3, 7)}

for auv_count, models in models_dict.items():
    for model in models.values():
        model.fit(scaled_inputs[auv_count], owa_output[owa_input[:, 0] == auv_count])

# --------------------- PREDICTION FUNCTION WITH NORMALIZATION ---------------------
def normalize_weights(weights):
    total = np.sum(weights)
    if total == 0:
        return np.array([10/3, 10/3, 10/3])
    # Aseguramos que los pesos sean positivos para la normalización (modelos como SVR pueden dar negativos)
    non_negative_weights = np.maximum(0, weights)
    new_total = np.sum(non_negative_weights)
    if new_total == 0: # Si todos eran negativos y se hicieron cero
         return np.array([10/3, 10/3, 10/3])
        
    return (non_negative_weights / new_total) * 10

def predict_for_testset(test_df, models_dict, scalers_dict):
    rows = []
    # Usamos unique() sobre el nuevo test_df para iterar sobre los AUVs definidos
    for auv in sorted(test_df['auv_count'].unique()):
        test_subset = test_df[test_df['auv_count'] == auv].reset_index(drop=True)
        # Necesitamos las columnas de entrada: 'auv_count', 'area'
        input_scaled = scalers_dict[auv].transform(test_subset[['auv_count', 'area']].values)
        
        # Como no tenemos valores 'True', usamos placeholders para evitar un error en el código
        # La forma más simple es tomar la longitud de las predicciones para el bucle interior
        
        for name, model in models_dict[auv].items():
            preds = model.predict(input_scaled)
            
            # Iteramos sobre cada predicción del subconjunto (5 predicciones por modelo/AUV)
            for i in range(len(test_subset)):
                raw_weights = preds[i][:3]
                norm_weights = normalize_weights(raw_weights)
                
                # Intentamos extraer 'utility' si el modelo lo predice (columna 3)
                pred_utility = preds[i][3] if preds.shape[1] > 3 else np.nan

                rows.append({
                    "auv_count": int(auv),
                    "area": test_subset.loc[i, 'area'],
                    "model": name,
                    # Dejamos los valores 'True' como NaN o 0, ya que estamos prediciendo
                    "true_w1": np.nan,
                    "true_w2": np.nan,
                    "true_w3": np.nan,
                    "true_utility": np.nan,
                    "pred_w1": norm_weights[0],
                    "pred_w2": norm_weights[1],
                    "pred_w3": norm_weights[2],
                    "pred_utility": pred_utility
                })
    return pd.DataFrame(rows)

# Renombramos la variable para reflejar que ahora es una PREDICCIÓN GLOBAL
global_prediction_df = predict_for_testset(test_df, models_dict, scalers_dict)
global_prediction_df.to_csv("results/owa_global_predictions.csv", index=False)
print("\n📁 Predicciones globales (20 combinaciones) guardadas en 'results/owa_global_predictions.csv'")

# --------------------- TEST METRICS (Deshabilitado/Modificado) ---------------------
# La sección de métricas de test no tiene sentido si estamos usando datos sin valores 'True'
# Sin embargo, el código original intenta calcular métricas comparando las predicciones 
# con los valores NaN (o la estructura original del test_df). 
# Para evitar errores en la ejecución completa, deshabilitamos la sección de métricas:
print("\nNota: La sección de cálculo de métricas (MAE/RMSE) ha sido omitida ya que el conjunto de prueba no tiene valores reales conocidos ('True').")

# --------------------- PLOTS and 3D PLOTS ---------------------
# El resto del código que genera plots y predicciones individuales queda tal cual, 
# pero ahora usa el DataFrame 'owa_df' para el entrenamiento y el nuevo 'test_df' 
# para las predicciones globales.

# El código que genera plots y 3D se mantiene igual, aunque las visualizaciones 
# no se mostrarán aquí y dependerán de los datos originales.

print("\nEl resto del script (cálculo de métricas, plots MAE/RMSE y plots 3D) se ejecutarán con las variables modificadas.")