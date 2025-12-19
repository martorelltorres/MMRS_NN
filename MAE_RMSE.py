import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# --- MOCK DATA AND TRAINING (Minimal structure for dependency satisfaction) ---
# ... (omitting training code for brevity, as we override metrics for plotting) ...

# --------------------- OVERRIDE METRICS WITH USER'S PLOT DATA ---------------------

# Data extracted from MAE.pdf and RMSE.pdf to recreate the plot exactly
data_for_plots = {
    '3 AUVs': {
        "SVR": {"MAE": 1.07, "RMSE": 1.15},
        "Random Forest": {"MAE": 1.18, "RMSE": 1.32},
        "Lasso": {"MAE": 1.18, "RMSE": 1.30},
        "Decision Tree": {"MAE": 1.19, "RMSE": 1.31},
        "Polynomial": {"MAE": 1.96, "RMSE": 2.08},
    },
    '4 AUVs': {
        "SVR": {"MAE": 1.30, "RMSE": 1.63},
        "Random Forest": {"MAE": 1.94, "RMSE": 2.27},
        "Lasso": {"MAE": 1.47, "RMSE": 1.78},
        "Decision Tree": {"MAE": 1.78, "RMSE": 2.13},
        "Polynomial": {"MAE": 1.83, "RMSE": 2.16},
    },
    '5 AUVs': {
        "SVR": {"MAE": 1.13, "RMSE": 1.39},
        "Random Forest": {"MAE": 1.35, "RMSE": 1.51},
        "Lasso": {"MAE": 1.19, "RMSE": 1.47},
        "Decision Tree": {"MAE": 1.26, "RMSE": 1.49},
        "Polynomial": {"MAE": 1.50, "RMSE": 1.59},
    },
    '6 AUVs': {
        "SVR": {"MAE": 0.99, "RMSE": 1.15},
        "Random Forest": {"MAE": 1.51, "RMSE": 2.05},
        "Lasso": {"MAE": 1.11, "RMSE": 1.24},
        "Decision Tree": {"MAE": 1.19, "RMSE": 1.30},
        "Polynomial": {"MAE": 2.40, "RMSE": 2.71},
    },
}

# Flatten the data structure for plotting
plot_data_list = []
for auv_group_name, models_data in data_for_plots.items():
    auv_count = int(auv_group_name.split()[0])
    for model_name, metrics in models_data.items():
        plot_data_list.append({
            "AUV Count": auv_count,
            "Group Name": auv_group_name,
            "Model": model_name,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"]
        })

plot_df = pd.DataFrame(plot_data_list)

# --- NEW SORTING LOGIC ---
# The user wants the RMSE plot sorted by ascending RMSE within each AUV group.
# 1. Sort by AUV Count (to group the data)
# 2. Sort by RMSE (to order within the group)
sorted_plot_df_rmse = plot_df.sort_values(by=["AUV Count", "RMSE"]).reset_index(drop=True)

# Define AUV colors (approximated from the provided PDF)
auv_colors = {
    3: '#fc8d59',  # Coral/Orange for 3 AUVS
    4: '#9970ab',  # Purple for 4 AUVS
    5: '#90b95c',  # Olive Green for 5 AUVS
    6: '#f4a5a5',  # Light Pink/Red for 6 AUVS
}


# --- PLOTTING LOGIC (Manual plt.bar to recreate complex layout) ---

# 1. MAE Plot (We keep the original sorting for MAE if not explicitly requested otherwise)
# Sort by MAE within each AUV group for the MAE plot
sorted_plot_df_mae = plot_df.sort_values(by=["AUV Count", "MAE"]).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(16, 7))
bar_width = 0.8
x_positions = np.arange(len(sorted_plot_df_mae))
mae_values = sorted_plot_df_mae['MAE'].values
auv_groups = sorted_plot_df_mae['AUV Count'].values
x_ticks_labels = sorted_plot_df_mae['Model'].values

# Plot bars and add value labels
for i in x_positions:
    mae = mae_values[i]
    auv = auv_groups[i]
    
    ax.bar(i, mae, color=auv_colors[auv], width=bar_width)
    ax.text(i, mae + 0.05, f'{mae:.2f}', ha='center', va='bottom', fontsize=10, color='black')

# Set X-axis labels to model names
ax.set_xticks(x_positions)
ax.set_xticklabels(x_ticks_labels, rotation=45, ha='right')

# Add legend patches manually for the AUV groups
legend_handles = [plt.Rectangle((0,0),1,1, fc=auv_colors[auv]) for auv in sorted(auv_colors.keys())]
legend_labels = [f'{auv} AUVs' for auv in sorted(auv_colors.keys())]
ax.legend(legend_handles, legend_labels, loc="upper left")

ax.set_title("MAE Comparison by Model Within Each AUV Group")
ax.set_ylabel("Mean Absolute Error (MAE)")
ax.set_xlabel("Regression Model")
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.ylim(0, 2.7)
plt.tight_layout()
plt.savefig("MAE_Comparison_Bar_Plot_Sorted_by_MAE.pdf")
plt.close(fig)

# 2. RMSE Plot (Sorted by RMSE within each AUV group as requested)
fig, ax = plt.subplots(figsize=(16, 7))
x_positions = np.arange(len(sorted_plot_df_rmse))
rmse_values = sorted_plot_df_rmse['RMSE'].values
auv_groups = sorted_plot_df_rmse['AUV Count'].values
x_ticks_labels = sorted_plot_df_rmse['Model'].values

# Plot bars and add value labels
for i in x_positions:
    rmse = rmse_values[i]
    auv = auv_groups[i]
    
    ax.bar(i, rmse, color=auv_colors[auv], width=bar_width)
    ax.text(i, rmse + 0.05, f'{rmse:.2f}', ha='center', va='bottom', fontsize=10, color='black')

# Set X-axis labels to model names
ax.set_xticks(x_positions)
ax.set_xticklabels(x_ticks_labels, rotation=45, ha='right')

# Add legend patches manually for the AUV groups
ax.legend(legend_handles, legend_labels, loc="upper left")

ax.set_title("RMSE Comparison by Model Within Each AUV Group")
ax.set_ylabel("Root Mean Squared Error (RMSE)")
ax.set_xlabel("Regression Model")
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.ylim(0, 3.0)
plt.tight_layout()
plt.savefig("RMSE_Comparison_Bar_Plot_Sorted_by_RMSE.pdf")
plt.close(fig)

print("\nPlots saved: MAE_Comparison_Bar_Plot_Sorted_by_MAE.png and RMSE_Comparison_Bar_Plot_Sorted_by_RMSE.png")