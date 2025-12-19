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
from sklearn.svm import SVR
from mpl_toolkits.mplot3d import Axes3D

# --------------------- CONFIGURATION FLAGS ---------------------
SHOW_PLOTS = False 

# --------------------- LOAD TRAINING DATA ---------------------
paths = [
    '/home/uib/MMRS_NN/weights/train/3AUV_optimal_weights.csv',
    '/home/uib/MMRS_NN/weights/train/4AUV_optimal_weights.csv',
    '/home/uib/MMRS_NN/weights/train/5AUV_optimal_weights.csv',
    '/home/uib/MMRS_NN/weights/train/6AUV_optimal_weights.csv',
]

# Load and concatenate training datasets
owa_df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
owa_input = owa_df[['auv_count', 'area']].values
owa_output = owa_df[['w1', 'w2', 'w3']].values

# Load test dataset
test_path = "/home/uib/MMRS_NN/results/owa_data/test/optimal_test_weights.csv"  
test_df = pd.read_csv(test_path)

# --------------------- SCALING ---------------------
scalers_dict = {}
scaled_inputs = {}

# Fit a specific scaler for each AUV count group
for auv in np.unique(owa_input[:, 0]):
    indices = owa_input[:, 0] == auv
    scaler = StandardScaler()
    scaled_inputs[auv] = scaler.fit_transform(owa_input[indices])
    scalers_dict[auv] = scaler

# --------------------- MODEL DEFINITIONS ---------------------
def create_models():
    # Consolidated SVR parameters based on previous optimization
    best_svr_params = {
        'kernel': 'rbf',
        'C': 0.5,
        'epsilon': 0.01,
        'gamma': 7
    }
    return {
        "Decision Tree": MultiOutputRegressor(DecisionTreeRegressor(max_depth=5, min_samples_leaf=3)),
        "Random Forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=1000)),
        "SVR": MultiOutputRegressor(SVR(**best_svr_params)),
        "Polynomial": MultiOutputRegressor(make_pipeline(PolynomialFeatures(4), LinearRegression())),
        "Lasso": MultiOutputRegressor(Lasso(alpha=0.3)),
    }

# Create a dictionary of models for each AUV count (3 to 6)
models_dict = {auv: create_models() for auv in range(3, 7)}

# Train models for each AUV group
for auv_count, models in models_dict.items():
    for model in models.values():
        mask = owa_input[:, 0] == auv_count
        model.fit(scaled_inputs[auv_count], owa_output[mask])

# --------------------- UTILITY FUNCTIONS ---------------------
def normalize_weights(weights):
    """Normalize weights to sum up to 1.0 (Probability Scale)"""
    total = np.sum(weights)
    if total <= 0:
        return np.array([1/3, 1/3, 1/3]) 
    return (weights / total)

def predict_for_testset(test_df, models_dict, scalers_dict):
    """Generates predictions for the test set across all models"""
    rows = []
    for auv in sorted(test_df['auv_count'].unique()):
        test_subset = test_df[test_df['auv_count'] == auv].reset_index(drop=True)
        input_data = test_subset[['auv_count', 'area']].values
        input_scaled = scalers_dict[auv].transform(input_data)
        
        true_output = test_subset[['w1', 'w2', 'w3', 'utility']].values

        for name, model in models_dict[auv].items():
            preds = model.predict(input_scaled)
            for i in range(len(test_subset)):
                # Extract and normalize predicted weights
                raw_weights = preds[i][:3]
                norm_weights = normalize_weights(raw_weights)
                # Prediction of utility if the model supports 4 outputs
                utility = preds[i][3] if preds.shape[1] > 3 else np.nan

                rows.append({
                    "auv_count": int(auv),
                    "area": test_subset.loc[i, 'area'],
                    "model": name,
                    "true_w1": true_output[i][0],
                    "true_w2": true_output[i][1],
                    "true_w3": true_output[i][2],
                    "true_utility": true_output[i][3],
                    "pred_w1": norm_weights[0],
                    "pred_w2": norm_weights[1],
                    "pred_w3": norm_weights[2],
                    "pred_utility": utility
                })
    return pd.DataFrame(rows)

# Execute predictions and save results
comparison_df = predict_for_testset(test_df, models_dict, scalers_dict)
comparison_df.to_csv("results/owa_model_predictions_on_test.csv", index=False)

# --------------------- TEST METRICS ---------------------
test_metrics_summary = []

for auv in sorted(test_df['auv_count'].unique()):
    test_subset = test_df[test_df['auv_count'] == auv]
    test_input = scalers_dict[auv].transform(test_subset[['auv_count', 'area']].values)
    test_output = test_subset[['w1', 'w2', 'w3']].values

    for name, model in models_dict[auv].items():
        preds = model.predict(test_input)
        norm_preds = np.apply_along_axis(normalize_weights, 1, preds[:, :3])
        
        mae = mean_absolute_error(test_output, norm_preds)
        rmse = mean_squared_error(test_output, norm_preds, squared=False)

        test_metrics_summary.append({
            "AUV Count": int(auv),
            "Model": name,
            "MAE": mae,
            "RMSE": rmse
        })

test_metrics_df = pd.DataFrame(test_metrics_summary)
test_metrics_df.to_csv("results/owa_model_test_metrics.csv", index=False)

# --------------------- VISUALIZATIONS ---------------------
if SHOW_PLOTS:
    # 1. Weights vs Exploration Area (Training Data)
    df_long = owa_df.melt(id_vars=["area", "auv_count"], 
                      value_vars=["w1", "w2", "w3"],
                      var_name="weight_type",
                      value_name="weight_value")

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_long, x="area", y="weight_value", hue="weight_type", marker='o')
    plt.title("w1, w2, w3 vs Exploration Area (Training Data)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 2. MAE Comparison - Ordered by Error within AUV Group
    plot_df_mae = test_metrics_df.copy().sort_values(by=["AUV Count", "MAE"], ascending=[True, True])
    plot_df_mae["Group"] = plot_df_mae["AUV Count"].astype(str) + " AUV - " + plot_df_mae["Model"]

    plt.figure(figsize=(16, 7))
    ax1 = sns.barplot(data=plot_df_mae, x="Group", y="MAE", palette="viridis", order=plot_df_mae["Group"])
    plt.title("MAE Comparison (Ordered by Error within AUV Group)", fontsize=15)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Absolute Error")

    # Add data labels for subtle differences
    for p in ax1.patches:
        ax1.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=9)
    plt.tight_layout()
    plt.show()

    # 3. RMSE Comparison - Ordered by Error within AUV Group
    plot_df_rmse = test_metrics_df.copy().sort_values(by=["AUV Count", "RMSE"], ascending=[True, True])
    plot_df_rmse["Group"] = plot_df_rmse["AUV Count"].astype(str) + " AUV - " + plot_df_rmse["Model"]

    plt.figure(figsize=(16, 7))
    ax2 = sns.barplot(data=plot_df_rmse, x="Group", y="RMSE", palette="magma", order=plot_df_rmse["Group"])
    plt.title("RMSE Comparison (Ordered by Error within AUV Group)", fontsize=15)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Root Mean Squared Error")

    for p in ax2.patches:
        ax2.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=9)
    plt.tight_layout()
    plt.show()

    # 4. 3D SVR Regression Surfaces
    X_3d = owa_df[['auv_count', 'area']].values
    y_3d = owa_df[['w1', 'w2', 'w3']].values

    auv_range = np.linspace(X_3d[:, 0].min(), X_3d[:, 0].max(), 30)
    area_range = np.linspace(X_3d[:, 1].min(), X_3d[:, 1].max(), 30)
    auv_grid, area_grid = np.meshgrid(auv_range, area_range)
    X_grid = np.c_[auv_grid.ravel(), area_grid.ravel()]

    fig = plt.figure(figsize=(18, 5))
    for i, weight_name in enumerate([r'$w_1$', r'$w_2$', r'$w_3$']):
        # Use consensus parameters for the 3D visual model
        model_3d = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=0.5, epsilon=0.01, gamma='scale'))
        model_3d.fit(X_3d, y_3d[:, i])
        y_pred_grid = model_3d.predict(X_grid).reshape(auv_grid.shape)

        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.plot_surface(auv_grid, area_grid, y_pred_grid, cmap='viridis', alpha=0.6)
        ax.scatter(X_3d[:, 0], X_3d[:, 1], y_3d[:, i], c='red', s=10)
        ax.set_xlabel('AUVs')
        ax.set_ylabel('Area [m²]')
        ax.set_zlabel(weight_name)
        ax.set_title(f'SVR Surface for {weight_name}')

    plt.tight_layout()
    plt.show()

print("\n Process completed. Results saved in the 'results/' folder.")