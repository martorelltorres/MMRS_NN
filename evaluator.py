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


