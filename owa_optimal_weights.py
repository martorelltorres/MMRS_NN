import pandas as pd

# Cargar el archivo CSV original
df = pd.read_csv("/home/uib/MMRS_NN/results/owa_data/test/owa_data.csv" )  

# Obtener los índices de las filas con la utilidad máxima por cada combinación de area y auv_count
idx_max_utility = df.groupby(['area', 'auv_count'])['utility'].idxmax()

# Extraer esas filas con sus respectivos w1, w2, w3 y utilidad
optimal_combinations = df.loc[idx_max_utility, ['area', 'auv_count', 'utility', 'w1', 'w2', 'w3']]

# Resetear índice para limpieza
optimal_combinations.reset_index(drop=True, inplace=True)

# Guardar en un nuevo archivo CSV
optimal_combinations.to_csv('/home/uib/MMRS_NN/results/owa_data/test/optimal_train_weights.csv', index=False)

print("Archivo 'optimal_train_weights.csv' guardado correctamente.")

# Cargar el archivo con las combinaciones óptimas

df = pd.read_csv('/home/uib/MMRS_NN/results/owa_data/test/optimal_train_weights.csv')

# Lista de cantidades de AUVs a procesar

auv_counts = [3, 4, 5, 6]

for auv in auv_counts:

    # Filtrar por número de AUVs

    df_auv = df[df['auv_count'] == auv]

    # Definir el nombre del archivo de salida

    output_filename = f'/home/uib/MMRS_NN/weights/test/{auv}AUV_optimal_weights.csv'

    # Guardar el CSV filtrado

    df_auv.to_csv(output_filename, index=False)

    print(f"Archivo '{auv}AUV_optimal_weights.csv' generado con éxito.")
