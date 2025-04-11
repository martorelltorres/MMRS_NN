import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import seaborn as sns
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

# --------------------- CARGA DE DATOS DE ENTRENAMIENTO ---------------------
# OWA
# owa_df = pd.read_csv('/home/antoni/MMRS_ws/src/MMRS_stack/MMRS_NN/data/utility_function/all_owa_data.csv')
# owa_df = pd.read_csv('/home/antoni/MMRS_ws/src/MMRS_stack/MMRS_NN/optimal_weights.csv')

paths = [
    '/home/antoni/MMRS_ws/src/MMRS_stack/MMRS_NN/weights/3AUV_weights.csv',
    '/home/antoni/MMRS_ws/src/MMRS_stack/MMRS_NN/weights/4AUV_weights.csv',
    '/home/antoni/MMRS_ws/src/MMRS_stack/MMRS_NN/weights/5AUV_weights.csv',
    '/home/antoni/MMRS_ws/src/MMRS_stack/MMRS_NN/weights/6AUV_weights.csv',
]

owa_df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
owa_input = owa_df[['auv_count', 'area']].values
owa_output = owa_df[['w1', 'w2', 'w3', 'utility']].values

# param_grid = {
#     'estimator__C': [0.1, 1, 10, 100],
#     'estimator__epsilon': [0.01, 0.1, 0.5],
#     'estimator__gamma': [0.001, 0.01, 0.1]
# }

# base_svr = svm.SVR(kernel='rbf')
# multi_svr = MultiOutputRegressor(base_svr)

# grid_search = GridSearchCV(multi_svr, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(owa_input, owa_output)

# print("Mejores hiperparámetros encontrados:", grid_search.best_params_)


# --------------------- DEFINICIÓN DE MODELOS ---------------------
owa_models = {
    "Decision Tree": MultiOutputRegressor(DecisionTreeRegressor(max_depth=4, min_samples_leaf=3)),
    "Random Forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=1000)),
    "SVM": MultiOutputRegressor(svm.SVR(kernel='rbf', C=10, epsilon=0.3, gamma =0.1)),

    # Más alto (C=10, C=100): el modelo se ajusta más a los datos (riesgo de overfitting).
    # Más bajo (C=0.1, C=0.01): permite más error en entrenamiento, buscando mejor generalización.
    # Valores pequeños (epsilon=0.01, 0.001): obligan al modelo a ajustarse más a los datos.
    # Valores grandes (epsilon=0.2, 0.5): hacen que el modelo sea menos sensible a pequeñas variaciones (más suave).
    # Bajo gamma (gamma=0.001, 0.01): decisiones más suaves, más generalización.
    # Alto gamma (gamma=0.1, 1, 10): el modelo se ajusta mucho a cada muestra (más riesgo de overfitting)

    "KNN": MultiOutputRegressor(neighbors.KNeighborsRegressor(n_neighbors=5)),
    "Poly Regression": MultiOutputRegressor(make_pipeline(PolynomialFeatures(4), LinearRegression())),
    "Ridge": MultiOutputRegressor(Ridge(alpha=0.5)),
    "Lasso": MultiOutputRegressor(Lasso(alpha=0.5)),
    "ElasticNet": MultiOutputRegressor(make_pipeline(StandardScaler(), ElasticNet(alpha=0.5)))
}


for model in owa_models.values():
    model.fit(owa_input, owa_output)

def find_optimal_owa_weights(auv_count, area, regressor):
    owa_combinations = [(10, 0, 0), (9, 1, 0), (8, 2, 0), (8, 2, 1), (8, 1, 1),
                        (7, 3, 0), (7, 2, 1), (6, 4, 0), (6, 2, 2), (6, 3, 1),
                        (5, 5, 0), (5, 4, 1), (5, 3, 2), (4, 4, 2), (4, 3, 3)]
    df = pd.DataFrame(owa_combinations, columns=['w1', 'w2', 'w3'])
    df['auv_count'] = auv_count
    df['area'] = area
    preds = regressor.predict(df[['auv_count', 'area']].values)
    best_idx = np.argmax(preds[:, 3])
    best = owa_combinations[best_idx]
    return preds[best_idx]

seen_areas = [10000, 20000, 30000, 40000, 50000, 60000]
intermediate_areas = [15000, 25000, 35000, 45000, 55000]
all_areas = seen_areas + intermediate_areas

unique_auv_counts = np.unique(owa_input[:, 0])

comparison_rows = []

for auv in unique_auv_counts:
    for area in all_areas:
        real_row = owa_df[(owa_df['auv_count'] == auv) & (owa_df['area'] == area)]
        if not real_row.empty:
            real_w1 = real_row['w1'].values[0]
            real_w2 = real_row['w2'].values[0]
            real_w3 = real_row['w3'].values[0]
        else:
            real_w1 = np.nan
            real_w2 = np.nan
            real_w3 = np.nan

        row_result = {
            "auv_count": auv,
            "area": area,
            "real_w1": real_w1,
            "real_w2": real_w2,
            "real_w3": real_w3,
        }

        for name, model in owa_models.items():

            utility_pred = find_optimal_owa_weights(auv, area, model)
            row_result[f"{name}_prediction"] = utility_pred[3]
            row_result[f"{name}_w1"] = utility_pred[0]
            row_result[f"{name}_w2"] = utility_pred[1]
            row_result[f"{name}_w3"] = utility_pred[2]

        comparison_rows.append(row_result)

# Convertir a DataFrame
comparison_df = pd.DataFrame(comparison_rows)

print(comparison_df.head())

# Guardar a CSV
comparison_df.to_csv("owa_model_predictions_all_areas.csv", index=False)
print("\n Comparativa guardada a 'owa_model_predictions_all_areas.csv'")

valid_rows = comparison_df.dropna(subset=["real_w1", "real_w2", "real_w3"])
detailed_error_summary = []

for auv in sorted(valid_rows['auv_count'].unique()):
    subset = valid_rows[valid_rows['auv_count'] == auv]
    
    for name in owa_models.keys():
        true_weights = subset[["real_w1", "real_w2", "real_w3"]].values
        pred_weights = subset[[f"{name}_w1", f"{name}_w2", f"{name}_w3"]].values

        mae = mean_absolute_error(true_weights, pred_weights)
        rmse = mean_squared_error(true_weights, pred_weights, squared=False)

        detailed_error_summary.append({
            "AUV Count": int(auv),
            "Model": name,
            "MAE": mae,
            "RMSE": rmse
        })

detailed_error_df = pd.DataFrame(detailed_error_summary)
detailed_error_df = detailed_error_df.sort_values(["AUV Count", "MAE"])

print("\nMétricas por dataset (auv_count) y modelo:")
print(detailed_error_df.round(3))

# Guardar a CSV
detailed_error_df.to_csv("owa_model_metrics_by_dataset.csv", index=False)
print("\n Métricas por dataset guardadas en 'owa_model_metrics_by_dataset.csv'")

# MAE
# Es el promedio de los errores absolutos entre las predicciones y los valores reales.
# Indica cuánto se desvían en promedio las predicciones de los valores reales.
# Se interpreta en las mismas unidades que los datos (por ejemplo, si tus pesos son enteros, MAE también será un número en esa escala).
# Penaliza por igual todos los errores, grandes o pequeños.

# RMSE
# Es la raíz cuadrada del promedio del error cuadrático. Penaliza más los errores grandes que pequeños.
# También se mide en las mismas unidades que tus datos.
# Da más peso a los errores grandes, lo que lo hace útil si te preocupa que tu modelo a veces falle mucho.
# A diferencia de MAE, RMSE no es robusto frente a outliers.


