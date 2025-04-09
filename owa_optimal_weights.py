import pandas as pd

# Cargar el archivo CSV original
df = pd.read_csv('/home/antoni/MMRS_ws/src/MMRS_stack/MMRS_NN/data/utility_function/all_owa_data.csv')  

# Obtener los índices de las filas con la utilidad máxima por cada combinación de area y auv_count
idx_max_utility = df.groupby(['area', 'auv_count'])['utility'].idxmax()

# Extraer esas filas con sus respectivos w1, w2, w3 y utilidad
optimal_combinations = df.loc[idx_max_utility, ['area', 'auv_count', 'utility', 'w1', 'w2', 'w3']]

# Resetear índice para limpieza
optimal_combinations.reset_index(drop=True, inplace=True)

# Guardar en un nuevo archivo CSV
optimal_combinations.to_csv('optimal_weights.csv', index=False)

print("Archivo 'optimal_weights.csv' guardado correctamente.")