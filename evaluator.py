import pandas as pd

# Cargar el CSV
data = pd.read_csv('/home/uib/MMRS_NN/data/new_areas/all_owa_data.csv', usecols=['utility', 'area', 'auv_count', 'w1', 'w2', 'w3'])

# Función para contar decimales
def count_decimals(x):
    s = str(x)
    if '.' in s:
        return len(s.split('.')[-1])
    else:
        return 0

# Agrupar por (area, auv_count)
grouped = data.groupby(['area', 'auv_count'])

results = []

for (area, auv_count), group in grouped:
    # Identificar la fila de predicción (w1, w2 o w3 con más de 2 decimales)
    pred_row = group[
        (group['w1'].apply(count_decimals) > 2) |
        (group['w2'].apply(count_decimals) > 2) |
        (group['w3'].apply(count_decimals) > 2)
    ]
    
    if pred_row.empty:
        print(f"No hay predicción encontrada para area={area}, auv_count={auv_count}")
        continue

    pred_row = pred_row.iloc[0]
    pred_utility = pred_row['utility']

    # Mejor y peor utilidad
    max_utility = group['utility'].max()
    min_utility = group['utility'].min()

    # Calcular el porcentaje normalizado
    if max_utility == min_utility:
        # Todas las utilidades son iguales
        score = 100.0
    else:
        score = (pred_utility - min_utility) / (max_utility - min_utility) * 100

    results.append({
        'area': area,
        'auv_count': auv_count,
        'pred_utility': pred_utility,
        'min_utility': min_utility,
        'max_utility': max_utility,
        'prediction_score(%)': round(score, 2)
    })

# Mostrar resultados individuales
for r in results:
    print(f"Área: {r['area']}, AUVs: {r['auv_count']}")
    print(f"Utilidad de la predicción: {r['pred_utility']:.6f}")
    print(f"Utilidad mínima: {r['min_utility']:.6f}")
    print(f"Utilidad máxima: {r['max_utility']:.6f}")
    print(f"Porcentaje de acierto normalizado: {r['prediction_score(%)']}%")
    print("-" * 50)

# Calcular y mostrar el porcentaje de acierto medio
if results:
    avg_score = sum(r['prediction_score(%)'] for r in results) / len(results)
    print(f"\nPorcentaje medio de acierto del sistema: {avg_score:.2f}%")
else:
    print("\nNo se encontraron predicciones para calcular el porcentaje medio.")


# Convertir los resultados a DataFrame para análisis agregado
results_df = pd.DataFrame(results)

# Promedio por área
print("\n--- Promedio de Acierto por Área ---")
area_means = results_df.groupby('area')['prediction_score(%)'].mean()
for area, score in area_means.items():
    print(f"Área: {area} -> Acierto medio: {score:.2f}%")

# Promedio por número de AUVs
print("\n--- Promedio de Acierto por Número de AUVs ---")
auv_means = results_df.groupby('auv_count')['prediction_score(%)'].mean()
for auv_count, score in auv_means.items():
    print(f"AUVs: {auv_count} -> Acierto medio: {score:.2f}%")

import matplotlib.pyplot as plt

# Datos
areas = [15000, 25000, 35000, 45000, 55000]
aciertos = [11.49, 100.00, 87.09, 84.31, 56.95]
labels = [f"{a:.0f} m²" for a in areas]  # Formato con unidad

# Colores uniformes
colors = ['cadetblue'] * len(areas)

# Crear gráfico
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, aciertos, color=colors)

# Títulos y etiquetas con tamaño de letra aumentado
plt.title('Average Prediction Accuracy by Exploration Area', fontsize=16)
plt.xlabel('Exploration Area [m²]', fontsize=14)
plt.ylabel('Average Prediction Accuracy [%]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 150)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar porcentaje sobre cada barra con fuente más grande
for i, val in enumerate(aciertos):
    plt.text(i, val + 2, f'{val:.1f}%', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Datos
data = {
    "Area": [15000]*4 + [25000]*4 + [35000]*4 + [45000]*4 + [55000]*4,
    "AUVs": [3, 4, 5, 6]*5,
    "Umin": [
        0.813, 0.900, 1.141, 1.222,
        0.983, 0.900, 1.008, 0.998,
        0.789, 0.763, 0.968, 1.011,
        1.033, 1.133, 1.101, 1.075,
        1.065, 1.065, 1.060, 1.057
    ],
    "Umax": [
        0.978, 1.010, 1.328, 1.322,
        1.022, 0.951, 1.088, 1.127,
        0.923, 1.186, 1.075, 1.296,
        1.116, 1.269, 1.194, 1.182,
        1.124, 1.171, 1.172, 1.154
    ],
    "Uwp": [
        0.835, 0.936, 1.141, 1.222,
        1.100, 0.956, 1.212, 1.169,
        0.885, 1.169, 1.100, 1.243,
        1.112, 1.215, 1.229, 1.162,
        1.104, 1.124, 1.066, 1.167
    ]
}

df = pd.DataFrame(data)

# Calcular la diferencia solo donde Uwp < Umax
# df["Diff"] = (df["Uwp"] - df["Umin"])/(df["Umax"]-df["Umin"])
# df["Diff"] = 100- (df["Uwp"]*100)/(df["Umax"])
# df["Diff"] = df["Diff"].apply(lambda x: x if x > 0 else 0)

# # Media por área
# mean_diff_by_area = df.groupby("Area")["Diff"].mean()
# print(mean_diff_by_area)

# areas = [15000, 25000, 35000, 45000, 55000]
# labels = [f"{a:.0f} m²" for a in areas]

# # Gráfico
# plt.figure(figsize=(8, 5))
# mean_diff_by_area.plot(kind="bar", color="cadetblue")
# plt.ylabel("Estimation error [%]", fontsize=14)
# plt.xlabel("Exploration Area [m²]", fontsize=14)
# plt.title("Mean Estimation Error per Exploration Area", fontsize=16)
# plt.xticks(rotation=45)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()


# Porcentajes de error medio por área ya calculados
# 15000    10.898508
# 25000     0.000000
# 35000     2.409976
# 45000     1.576447
# 55000     3.709348

errores_porcentuales = [10.89, 0.00, 2.40, 1.57, 3.70]
areas = [15000, 25000, 35000, 45000, 55000]
labels = [f"{a:.0f} m²" for a in areas]

# Colores personalizados
colors = ['cadetblue'] * len(areas)

# Crear gráfico
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, errores_porcentuales, color=colors)
plt.title('Average Prediction Error by Exploration Area', fontsize=16)
plt.xlabel('Exploration Area [m²]', fontsize=14)
plt.ylabel('Average Prediction Error [%]', fontsize=14)
plt.ylim(0, max(errores_porcentuales) + 5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar porcentaje sobre cada barra
for i, val in enumerate(errores_porcentuales):
    plt.text(i, val + 1, f'{val:.2f}%', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()

# Datos originales
data = {
    "Area": [15000]*4 + [25000]*4 + [35000]*4 + [45000]*4 + [55000]*4,
    "AUVs": [3, 4, 5, 6]*5,
    "Umin": [
        0.813, 0.900, 1.141, 1.222,
        0.983, 0.900, 1.008, 0.998,
        0.789, 0.763, 0.968, 1.011,
        1.033, 1.133, 1.101, 1.075,
        1.065, 1.065, 1.060, 1.057
    ],
    "Umax": [
        0.978, 1.010, 1.328, 1.322,
        1.022, 0.951, 1.088, 1.127,
        0.923, 1.186, 1.075, 1.296,
        1.116, 1.269, 1.194, 1.182,
        1.124, 1.171, 1.172, 1.154
    ],
    "Uwp": [
        0.835, 0.936, 1.141, 1.222,
        1.100, 0.956, 1.212, 1.169,
        0.885, 1.169, 1.100, 1.243,
        1.112, 1.215, 1.229, 1.162,
        1.104, 1.124, 1.066, 1.167
    ]
}

# Crear DataFrame
df = pd.DataFrame(data)

# Filtrar los casos donde Uwp > Umax
df_mejora = df[df["Uwp"] > df["Umax"]].copy()

# Calcular la mejora porcentual
df_mejora["Mejora_%"] = (df_mejora["Uwp"] - df_mejora["Umax"]) / df_mejora["Umax"] * 100

# Agrupar por área y calcular la media de mejora
mejora_por_area = df_mejora.groupby("Area")["Mejora_%"].mean().reset_index()

# Mostrar resultado
print(mejora_por_area)

# 0  25000  5.820406
# 1  35000  2.325581
# 2  45000  2.931323
# 3  55000  1.126516

errores_porcentuales = [0, 5.82, 2.32, 2.93, 1.12]
areas = [15000, 25000, 35000, 45000, 55000]
labels = [f"{a:.0f} m²" for a in areas]
colors = ['cadetblue'] * len(areas)

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, errores_porcentuales, color=colors)

# Aumentar tamaños de fuente
plt.title('Average Utility Improvement by Exploration Area', fontsize=16)
plt.xlabel('Exploration Area [m²]', fontsize=14)
plt.ylabel('Average Utility Improvement [%]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, max(errores_porcentuales) + 3)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar porcentaje sobre cada barra con fuente más grande
for i, val in enumerate(errores_porcentuales):
    plt.text(i, val + 0.5, f'{val:.2f}%', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()




