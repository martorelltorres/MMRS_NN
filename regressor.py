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
SHOW_PLOTS = True  

# --------------------- LOAD TRAINING DATA ---------------------
paths = [
    '/home/uib/MMRS_NN/weights/train/3AUV_optimal_weights.csv',
    '/home/uib/MMRS_NN/weights/train/4AUV_optimal_weights.csv',
    '/home/uib/MMRS_NN/weights/train/5AUV_optimal_weights.csv',
    '/home/uib/MMRS_NN/weights/train/6AUV_optimal_weights.csv',
]

owa_df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
owa_input = owa_df[['auv_count', 'area']].values
# FIX 1: Include 'utility' in the training output (4 columns total)
owa_output = owa_df[['w1', 'w2', 'w3', 'utility']].values

test_path = "/home/uib/MMRS_NN/results/owa_data/test/optimal_test_weights.csv"  
test_df = pd.read_csv(test_path)

# --------------------- SCALING ---------------------
scalers_dict = {}
scaled_inputs = {}

for auv in np.unique(owa_input[:, 0]):
    indices = owa_input[:, 0] == auv
    scaler = StandardScaler()
    scaled_inputs[auv] = scaler.fit_transform(owa_input[indices])
    scalers_dict[auv] = scaler

# --------------------- MODEL DEFINITIONS ---------------------
def create_models():
    # Consensus SVR parameters
    best_svr_params = {'kernel': 'rbf', 'C': 0.5, 'epsilon': 0.01, 'gamma': 'scale'}
    
    return {
        # FIX 2: Increased depth for Decision Tree to capture subtle utility changes
        "Decision Tree": MultiOutputRegressor(DecisionTreeRegressor(max_depth=10, min_samples_leaf=1)),
        "Random Forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=1000)),
        "SVR": MultiOutputRegressor(SVR(**best_svr_params)),
        "Polynomial": MultiOutputRegressor(make_pipeline(PolynomialFeatures(4), LinearRegression())),
        "Lasso": MultiOutputRegressor(Lasso(alpha=0.01)), # Reduced alpha to allow more flexibility
    }

models_dict = {auv: create_models() for auv in range(3, 7)}

# Train models
for auv_count, models in models_dict.items():
    mask = owa_input[:, 0] == auv_count
    for model in models.values():
        model.fit(scaled_inputs[auv_count], owa_output[mask])

# --------------------- UTILITY FUNCTIONS ---------------------
def normalize_weights(weights):
    total = np.sum(weights)
    if total <= 0: return np.array([1/3, 1/3, 1/3])
    return (weights / total)

def predict_for_testset(test_df, models_dict, scalers_dict):
    rows = []
    for auv in sorted(test_df['auv_count'].unique()):
        test_subset = test_df[test_df['auv_count'] == auv].reset_index(drop=True)
        input_scaled = scalers_dict[auv].transform(test_subset[['auv_count', 'area']].values)
        
        # Real values from test file
        true_output = test_subset[['w1', 'w2', 'w3', 'utility']].values

        for name, model in models_dict[auv].items():
            preds = model.predict(input_scaled)
            for i in range(len(test_subset)):
                # FIX 3: Correctly indexing the 4 outputs
                raw_weights = preds[i][:3]
                norm_weights = normalize_weights(raw_weights)
                pred_utility = preds[i][3] # This will now vary

                rows.append({
                    "auv_count": int(auv), "area": test_subset.loc[i, 'area'], "model": name,
                    "true_w1": true_output[i][0], "true_w2": true_output[i][1], "true_w3": true_output[i][2],
                    "true_utility": true_output[i][3],
                    "pred_w1": norm_weights[0], "pred_w2": norm_weights[1], "pred_w3": norm_weights[2],
                    "pred_utility": pred_utility
                })
    return pd.DataFrame(rows)

# Save predictions
comparison_df = predict_for_testset(test_df, models_dict, scalers_dict)
comparison_df.to_csv("results/owa_model_predictions_on_test.csv", index=False)

# --------------------- TEST METRICS ---------------------
test_metrics_summary = []
for auv in sorted(test_df['auv_count'].unique()):
    test_subset = test_df[test_df['auv_count'] == auv]
    test_input = scalers_dict[auv].transform(test_subset[['auv_count', 'area']].values)
    # We evaluate error on the 3 weights
    test_output_weights = test_subset[['w1', 'w2', 'w3']].values

    for name, model in models_dict[auv].items():
        preds = model.predict(test_input)
        norm_preds = np.apply_along_axis(normalize_weights, 1, preds[:, :3])
        
        mae = mean_absolute_error(test_output_weights, norm_preds)
        rmse = mean_squared_error(test_output_weights, norm_preds, squared=False)

        test_metrics_summary.append({
            "AUV Count": int(auv), "Model": name, "MAE": mae, "RMSE": rmse
        })

test_metrics_df = pd.DataFrame(test_metrics_summary)
test_metrics_df.to_csv("results/owa_model_test_metrics.csv", index=False)

# --------------------- VISUALIZATIONS ---------------------
if SHOW_PLOTS:
    # Barplot MAE
    plot_df_mae = test_metrics_df.copy().sort_values(by=["AUV Count", "MAE"])
    plot_df_mae["Group"] = plot_df_mae["AUV Count"].astype(str) + " AUV - " + plot_df_mae["Model"]
    plt.figure(figsize=(16, 7))
    ax = sns.barplot(data=plot_df_mae, x="Group", y="MAE", palette="viridis", order=plot_df_mae["Group"])
    plt.xticks(rotation=45, ha="right")
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=8)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(16, 8))
    
    # Create a 'Group' label for the X axis
    comparison_df["Group_Area"] = comparison_df["auv_count"].astype(str) + "AUV\n" + comparison_df["area"].astype(str)
    
    # Melt the dataframe to have 'true_utility' and 'pred_utility' in the same column for seaborn
    utility_plot_df = comparison_df.melt(
        id_vars=["Group_Area", "model"], 
        value_vars=["true_utility", "pred_utility"],
        var_name="Utility_Type", 
        value_name="Value"
    )

    # Plotting
    ax = sns.barplot(
        data=utility_plot_df, 
        x="Group_Area", 
        y="Value", 
        hue="Utility_Type", 
        palette={"true_utility": "#2ecc71", "pred_utility": "#e74c3c"}
    )
    
    plt.title("Comparison: True Utility vs Predicted Utility (Across all Models)", fontsize=16)
    plt.ylabel("Utility Value", fontsize=12)
    plt.xlabel("Configuration (AUV Count & Area)", fontsize=12)
    plt.legend(title="Utility Type")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Adjusting layout to prevent label overlap
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Optional: Error Distribution Plot for Utility
    plt.figure(figsize=(10, 6))
    comparison_df['utility_error'] = comparison_df['true_utility'] - comparison_df['pred_utility']
    sns.histplot(data=comparison_df, x="utility_error", hue="model", kde=True, element="step")
    plt.title("Utility Prediction Error Distribution by Model")
    plt.xlabel("Error (True - Predicted)")
    plt.axvline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.show()

print("\n Process completed. Predictions now include dynamic utility values.")