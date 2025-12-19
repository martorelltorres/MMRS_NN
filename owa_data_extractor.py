#!/usr/bin/env python3
import rosbag
import csv
import os
import numpy as np
import pandas as pd
import yaml

# Define paths and parameters
base_path = "/home/uib/MRS_data/simulation_data/test_data"
# areas = [10000,20000,30000,40000,50000,60000]
areas = [15000,25000,35000,45000,55000]
auv_counts = ["3AUVs", "4AUVs", "5AUVs", "6AUVs"]
aggregation_methods = ["owa"]

# Topics of interest
topics_of_interest = [
    "/mrs/allocator_communication_latency",
    "/mrs/asv_travelled_distance",
    "/mrs/allocator_data_transmited",
    "/mrs/allocator_data_buffered",
    "/mrs/asv0_priority_communication_latency",
    "/mrs/asv0_regular_communication_latency",
    "/mrs/aggregation_model_info"
]

# Parameter combinations
parameter_combinations_map = {
    "artm": [[0, 10], [2.5, 7.5], [5, 5], [7.5, 2.5], [10, 0]]
}

# Storage for processed data
owa_data = []
response_threshold_data = []

# Main loop
for area in areas:
    area_data = []

    for auv_count in auv_counts:
        for method in aggregation_methods:
            bagfile_path = os.path.join(base_path, str(area), auv_count, method, "bagfiles")
            if not os.path.exists(bagfile_path):
                print(f"[!] Carpeta no encontrada: {bagfile_path}")
                continue

            print(f"\nProcesando: {bagfile_path}")
            bag_files = [os.path.join(bagfile_path, f) for f in os.listdir(bagfile_path) if f.endswith('.bag')]
            print(f"→ {len(bag_files)} archivos .bag encontrados")

            parameter_combinations = parameter_combinations_map.get(method, [])
            folder_data = []

            for bag_index, bag_path in enumerate(bag_files):
                try:
                    bag = rosbag.Bag(bag_path)
                except Exception as e:
                    print(f"[!] Error abriendo bag: {bag_path}\n{e}")
                    continue

                reg_latency_values, prior_latency_values = np.array([]), np.array([])

                sum_data, sum_reg_objects, sum_prior_objects, travelled_distance = 0, 0, 0, 0

                # Parámetros
                if method == "owa":
                    yaml_path = os.path.join(base_path, str(area), auv_count, method, "params", f"params_{bag_index}.yaml")
                    try:
                        with open(yaml_path, 'r') as yaml_file:
                            yaml_data = yaml.safe_load(yaml_file)
                            w1 = yaml_data.get('w1', 0)
                            w2 = yaml_data.get('w2', 0)
                            w3 = yaml_data.get('w3', 0)
                            param = [w1, w2, w3]
                    except Exception as e:
                        print(f"[!] Error leyendo YAML: {yaml_path} → {e}")
                        param = [0, 0, 0]
                else:
                    param = parameter_combinations[bag_index % len(parameter_combinations)]

                for topic, msg, t in bag.read_messages(topics=topics_of_interest):
                    if topic.endswith("regular_communication_latency"):
                        reg_latency = sum(getattr(msg, 'comm_latency', (0,) * 6))
                        reg_latency_values = np.append(reg_latency_values, reg_latency)

                    if topic.endswith("priority_communication_latency"):
                        prior_latency = sum(getattr(msg, 'comm_latency', (0,) * 6))
                        prior_latency_values = np.append(prior_latency_values, prior_latency)

                    if topic.endswith("travelled_distance"):
                        travelled_distance = msg.travelled_distance

                    if topic.endswith("allocator_data_transmited"):
                        sum_data = sum(getattr(msg, 'transmitted_data', (0,) * 6))
                        sum_reg_objects = sum(getattr(msg, 'transmitted_regular_objects', (0,) * 6))
                        sum_prior_objects = sum(getattr(msg, 'transmitted_priority_objects', (0,) * 6))

                bag.close()

                reg_latency = np.mean(reg_latency_values) if len(reg_latency_values) > 0 else 0
                reg_latency_std = np.std(reg_latency_values, ddof=1) if len(reg_latency_values) > 1 else 0
                prior_latency = np.mean(prior_latency_values) if len(prior_latency_values) > 0 else 0
                prior_latency_std = np.std(prior_latency_values, ddof=1) if len(prior_latency_values) > 1 else 0

                # Datos comunes
                row = {
                    'area': area,
                    'auv_count': int(auv_count.replace('AUVs', '')),
                    'method': method,
                    'regular_latency': reg_latency,
                    'priority_latency': prior_latency,
                    'regular_std': reg_latency_std,
                    'priority_std': prior_latency_std,
                    'transmitted_data': sum_data,
                    'priority_objects': sum_prior_objects,
                    'regular_objects': sum_reg_objects,
                    'travelled_distance': travelled_distance
                }

                # Añadir parámetros
                if method == "owa":
                    row.update({'w1': param[0], 'w2': param[1], 'w3': param[2]})
                elif method == "artm":
                    row.update({'a': param[0], 'b': param[1]})

                folder_data.append(row)

            # Guardar los datos del área
            area_data.extend(folder_data)

    # Convertir todos los datos del área en DataFrame
    df_area = pd.DataFrame(area_data)

    # Normalizar dentro del área
    def normalize_column(df, column):
        min_val = df[column].min()
        max_val = df[column].max()
        if pd.isna(min_val) or pd.isna(max_val) or max_val - min_val == 0:
            print(f"[!] No se puede normalizar la columna {column} para área {area}: min={min_val}, max={max_val}")
            df[column + '_normalized'] = np.nan
        else:
            df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)

    def normalize_div_max(df, column):
        max_val = df[column].max()
        if pd.isna(max_val) or max_val == 0:
            print(f"[!] No se puede normalizar la columna {column} con división por max en área {area}")
            df[column + '_normalized'] = np.nan
        else:
            df[column + '_normalized'] = df[column] / max_val

    for col in ['regular_latency', 'priority_latency', 'regular_std', 'priority_std', 'travelled_distance']:
        normalize_column(df_area, col)

    for col in ['priority_objects', 'regular_objects', 'transmitted_data']:
        normalize_div_max(df_area, col)

    # Calcular utilidad
    alpha = beta = gamma = 1
    df_area['priority'] = df_area['priority_objects_normalized'] / (1 + np.exp((df_area['priority_latency_normalized'] + df_area['priority_std_normalized'])))
    df_area['regular'] = df_area['regular_objects_normalized'] / (1 + np.exp((df_area['regular_latency_normalized'] + df_area['regular_std_normalized'])))
    df_area['distance'] = np.exp(-df_area['travelled_distance_normalized'])
    df_area['utility'] = df_area['priority'] + 0.5*df_area['regular'] + df_area['distance']

    # Guardar según el método
    owa_data.extend(df_area[df_area['method'] == 'owa'].to_dict('records'))
    response_threshold_data.extend(df_area[df_area['method'] == 'artm'].to_dict('records'))

# Guardar CSV finales
pd.DataFrame(owa_data).to_csv(os.path.join("/home/uib/MMRS_NN/results/owa_data/test", "owa_data.csv"), index=False)
pd.DataFrame(response_threshold_data).to_csv(os.path.join("/home/uib/MMRS_NN/results/owa_data/test", "artm_data.csv"), index=False)

print("\n✅ Extracción y normalización por área finalizadas. CSVs generados.")
