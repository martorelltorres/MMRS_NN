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

# --------------------- CARGA DE DATOS DE ENTRENAMIENTO ---------------------
# OWA
# owa_df = pd.read_csv('/home/antoni/MMRS_ws/src/MMRS_stack/MMRS_NN/data/utility_function/all_owa_data.csv')
owa_df = pd.read_csv('/home/antoni/MMRS_ws/src/MMRS_stack/MMRS_NN/optimal_weights.csv')
owa_input = owa_df[['auv_count', 'area']].values
owa_output = owa_df[['w1', 'w2', 'w3', 'utility']].values


# --------------------- DEFINICIÓN DE MODELOS ---------------------
owa_models = {
    "Decision Tree": MultiOutputRegressor(DecisionTreeRegressor(max_depth=7)),
    "Random Forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=1000)),
    "SVM": MultiOutputRegressor(svm.SVR(kernel='rbf', C=1.0, epsilon=0.1)),
    "KNN": MultiOutputRegressor(neighbors.KNeighborsRegressor(n_neighbors=5)),
    "Poly Regression": MultiOutputRegressor(make_pipeline(PolynomialFeatures(4), LinearRegression()))
}

# Entrenamiento
for model in owa_models.values():
    model.fit(owa_input, owa_output)

# --------------------- FUNCIONES DE BÚSQUEDA ÓPTIMA ---------------------
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

# --------------------- COMPARATIVA DE RESULTADOS COMPLETA (ÁREAS VISTAS + INTERMEDIAS) ---------------------

# Áreas del entrenamiento (vistas) y nuevas áreas intermedias (no vistas)
seen_areas = [10000, 20000, 30000, 40000, 50000, 60000]
intermediate_areas = [15000, 25000, 35000, 45000, 55000]
all_areas = seen_areas + intermediate_areas

# Extraer valores únicos de auv_count del conjunto original
unique_auv_counts = np.unique(owa_input[:, 0])

# Crear combinaciones de (auv_count, area)
comparison_rows = []

for auv in unique_auv_counts:
    for area in all_areas:
        # Buscar los valores reales si existen
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

# Mostrar ejemplo
print(comparison_df.head())

# Guardar a CSV
comparison_df.to_csv("owa_model_predictions_all_areas.csv", index=False)
print("\n✅ Comparativa guardada en 'owa_model_predictions_all_areas.csv'")


