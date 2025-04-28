import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
import seaborn as sns
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor

# --------------------- CARGA DE DATOS DE ENTRENAMIENTO ---------------------
paths = [
    '/home/uib/MMRS_NN/weights/3AUV_weights.csv',
    '/home/uib/MMRS_NN/weights/4AUV_weights.csv',
    '/home/uib/MMRS_NN/weights/5AUV_weights.csv',
    '/home/uib/MMRS_NN/weights/6AUV_weights.csv',
]

owa_df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
owa_input = owa_df[['auv_count', 'area']].values
owa_output = owa_df[['w1', 'w2', 'w3', 'utility']].values

test_path = "/home/uib/MMRS_NN/weights/optimal_test_weights.csv"  
test_df = pd.read_csv(test_path)

# --------------------- ESCALADO ---------------------
scalers_dict = {}
scaled_inputs = {}

for auv in np.unique(owa_input[:, 0]):
    indices = owa_input[:, 0] == auv
    scaler = StandardScaler()
    scaled_inputs[auv] = scaler.fit_transform(owa_input[indices])
    scalers_dict[auv] = scaler

# --------------------- DEFINICIÓN DE MODELOS ---------------------
def create_models():
    return {
        "Decision Tree": MultiOutputRegressor(DecisionTreeRegressor(max_depth=5, min_samples_leaf=3)),
        "Random Forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=1000)),
        "SVM": MultiOutputRegressor(svm.SVR(kernel='rbf', C=10, epsilon=0.3, gamma=0.1)),
        "KNN": MultiOutputRegressor(neighbors.KNeighborsRegressor(n_neighbors=5)),
        "Poly Regression": MultiOutputRegressor(make_pipeline(PolynomialFeatures(4), LinearRegression())),
        "Ridge": MultiOutputRegressor(Ridge(alpha=0.5)),
        "Lasso": MultiOutputRegressor(Lasso(alpha=0.5)),
        "ElasticNet": MultiOutputRegressor(make_pipeline(StandardScaler(), ElasticNet(alpha=0.5)))
    }

models_dict = {auv: create_models() for auv in range(3, 7)}

for auv_count, models in models_dict.items():
    for model in models.values():
        model.fit(scaled_inputs[auv_count], owa_output[owa_input[:, 0] == auv_count])

# --------------------- PREDICCIÓN ---------------------
def predict_for_testset(test_df, models_dict, scalers_dict):
    rows = []
    for auv in sorted(test_df['auv_count'].unique()):
        test_subset = test_df[test_df['auv_count'] == auv].reset_index(drop=True)
        input_scaled = scalers_dict[auv].transform(test_subset[['auv_count', 'area']].values)
        true_output = test_subset[['w1', 'w2', 'w3', 'utility']].values

        for name, model in models_dict[auv].items():
            preds = model.predict(input_scaled)
            for i in range(len(test_subset)):
                rows.append({
                    "auv_count": int(auv),
                    "area": test_subset.loc[i, 'area'],
                    "model": name,
                    "true_w1": true_output[i][0],
                    "true_w2": true_output[i][1],
                    "true_w3": true_output[i][2],
                    "true_utility": true_output[i][3],
                    "pred_w1": preds[i][0],
                    "pred_w2": preds[i][1],
                    "pred_w3": preds[i][2],
                    "pred_utility": preds[i][3] if preds.shape[1] > 3 else np.nan  # seguridad por si solo predice w1-w3
                })
    return pd.DataFrame(rows)

comparison_df = predict_for_testset(test_df, models_dict, scalers_dict)
print(comparison_df.head())
comparison_df.to_csv("results/owa_model_predictions_on_test.csv", index=False)
print("\n📁 Predicciones sobre el conjunto de test guardadas en 'results/owa_model_predictions_on_test.csv'")


# --------------------- MÉTRICAS TEST ---------------------
test_metrics_summary = []

for auv in sorted(test_df['auv_count'].unique()):
    test_subset = test_df[test_df['auv_count'] == auv]
    test_input = scalers_dict[auv].transform(test_subset[['auv_count', 'area']].values)
    test_output = test_subset[['w1', 'w2', 'w3']].values

    for name, model in models_dict[auv].items():
        preds = model.predict(test_input)
        mae = mean_absolute_error(test_output, preds[:, :3])
        rmse = mean_squared_error(test_output, preds[:, :3], squared=False)

        test_metrics_summary.append({
            "AUV Count": int(auv),
            "Model": name,
            "MAE": mae,
            "RMSE": rmse
        })

test_metrics_df = pd.DataFrame(test_metrics_summary).sort_values(["AUV Count", "MAE"])
print("\n🔍 Métricas sobre el conjunto de test:")
print(test_metrics_df.round(3))
test_metrics_df.to_csv("results/owa_model_test_metrics.csv", index=False)
print("\n📁 Métricas de test guardadas en 'results/owa_model_test_metrics.csv'")

# --------------------- GRAFICADO ---------------------
df_long = owa_df.melt(id_vars=["area", "auv_count", "utility"], 
                  value_vars=["w1", "w2", "w3"],
                  var_name="weight_type",
                  value_name="weight_value")

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_long, x="area", y="weight_value", hue="weight_type", marker='o')
plt.title("w1, w2, w3 vs exploration area")
plt.xlabel("Exploration area")
plt.ylabel("Weights value")
plt.grid(True)
plt.legend(title="Weights")
plt.tight_layout()
plt.show()

# Crear una columna auxiliar combinando AUV y modelo
test_metrics_df["Group"] = test_metrics_df["AUV Count"].astype(str) + " AUV - " + test_metrics_df["Model"]

# Ordenar por AUV Count y luego por MAE dentro de cada grupo
sorted_df = test_metrics_df.sort_values(["AUV Count", "MAE"])

# Crear un orden personalizado para las barras
custom_order = sorted_df["Group"].values

# Gráfico de barras con orden personalizado
plt.figure(figsize=(16, 6))
sns.barplot(
    data=sorted_df,
    x="Group",
    y="MAE",
    palette="viridis",
    order=custom_order
)

plt.title("Comparación de MAE por modelo ordenado dentro de cada grupo de AUVs")
plt.ylabel("Mean Absolute Error (MAE)")
plt.xlabel("Modelo por grupo de AUVs")
plt.xticks(rotation=45, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

auv_count=6
area=35000

if auv_count not in models_dict:
    print(f"No hay modelos entrenados para {auv_count} AUVs.")

input_data = np.array([[auv_count, area]])
input_scaled = scalers_dict[auv_count].transform(input_data)

print(f"\n📍 Predicción para {auv_count} AUVs y área = {area}:\n")
for model_name, model in models_dict[auv_count].items():
    prediction = model.predict(input_scaled)[0]
    w1, w2, w3, utility = prediction
    print(f"🔹 {model_name}:")
    print(f"    w1 = {w1:.3f}, w2 = {w2:.3f}, w3 = {w3:.3f}, utility = {utility:.3f}")

from scipy.spatial import distance

def optimize_predicted_weights(pred_w, known_weights_utilities, top_k=10, utility_threshold=0.9):
    """
    Estrategia mejorada:
    - Buscar los top_k pesos más cercanos.
    - Si no se alcanza buena utilidad, buscar en todo el conjunto.
    
    pred_w: array (3,)
    known_weights_utilities: DataFrame ['w1', 'w2', 'w3', 'utility']
    top_k: número de candidatos cercanos a considerar.
    utility_threshold: proporción mínima respecto a la mejor utilidad (por ejemplo, 0.9)
    """
    weights = known_weights_utilities[['w1', 'w2', 'w3']].values
    utilities = known_weights_utilities['utility'].values

    # Distancias euclídeas
    dists = distance.cdist([pred_w], weights, 'euclidean')[0]

    # Top_k más cercanos
    top_k_indices = np.argsort(dists)[:top_k]
    best_in_top_k_idx = top_k_indices[np.argmax(utilities[top_k_indices])]
    best_in_top_k_util = utilities[best_in_top_k_idx]

    # Mejor utilidad absoluta en todo el dataset
    global_best_util = np.max(utilities)

    # Si la mejor del top_k alcanza al menos el umbral, nos la quedamos
    if best_in_top_k_util >= utility_threshold * global_best_util:
        final_idx = best_in_top_k_idx
    else:
        # Si no, buscamos el mejor de todo el conjunto
        global_best_idx = np.argmax(utilities)
        final_idx = global_best_idx

    return weights[final_idx], utilities[final_idx]

def predict_for_testset_with_optimization(test_df, models_dict, scalers_dict, owa_df):
    rows = []
    for auv in sorted(test_df['auv_count'].unique()):
        test_subset = test_df[test_df['auv_count'] == auv].reset_index(drop=True)
        input_scaled = scalers_dict[auv].transform(test_subset[['auv_count', 'area']].values)
        true_output = test_subset[['w1', 'w2', 'w3', 'utility']].values
        
        # Dataset real filtrado para ese auv_count
        known_weights_utilities = owa_df[owa_df['auv_count'] == auv]

        for name, model in models_dict[auv].items():
            preds = model.predict(input_scaled)

            for i in range(len(test_subset)):
                pred_w = preds[i][:3]
                pred_util = preds[i][3] if preds.shape[1] > 3 else np.nan

                # Optimización
                opt_w, opt_util = optimize_predicted_weights(pred_w, known_weights_utilities)

                rows.append({
                    "auv_count": int(auv),
                    "area": test_subset.loc[i, 'area'],
                    "model": name,
                    "true_w1": true_output[i][0],
                    "true_w2": true_output[i][1],
                    "true_w3": true_output[i][2],
                    "true_utility": true_output[i][3],
                    "pred_w1": pred_w[0],
                    "pred_w2": pred_w[1],
                    "pred_w3": pred_w[2],
                    "pred_utility": pred_util,
                    "opt_w1": opt_w[0],
                    "opt_w2": opt_w[1],
                    "opt_w3": opt_w[2],
                    "opt_utility": opt_util
                })
    return pd.DataFrame(rows)

def plot_predicted_vs_optimized_utilities(comparison_df):
    plt.figure(figsize=(8, 6))
    
    # Dibujamos la línea y = x como referencia
    plt.plot([0, 1], [0, 1], 'k--', label='Ideal (sin mejora)')
    
    # Dibujamos puntos: cada modelo puede tener su color
    models = comparison_df['model'].unique()
    for model in models:
        subset = comparison_df[comparison_df['model'] == model]
        plt.scatter(subset['pred_utility'], subset['opt_utility'], label=model, alpha=0.6)
    
    plt.xlabel('Utilidad predicha')
    plt.ylabel('Utilidad optimizada')
    plt.title('Comparativa utilidad predicha vs utilidad optimizada')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

comparison_df = predict_for_testset_with_optimization(test_df, models_dict, scalers_dict, owa_df)
comparison_df.to_csv("results/owa_model_predictions_and_optimization.csv", index=False)
plot_predicted_vs_optimized_utilities(comparison_df)
