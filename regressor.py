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
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D

# --------------------- LOAD TRAINING DATA ---------------------
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
    return {
        "Decision Tree": MultiOutputRegressor(DecisionTreeRegressor(max_depth=5, min_samples_leaf=3)),
        "Random Forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=1000)),
        "SVR": MultiOutputRegressor(svm.SVR(kernel='rbf', C=7, epsilon=1.2, gamma=0.1)),
        "Polynomial" : MultiOutputRegressor(make_pipeline(PolynomialFeatures(4), LinearRegression())),
        "Lasso": MultiOutputRegressor(Lasso(alpha=0.5)),
    }

models_dict = {auv: create_models() for auv in range(3, 7)}

for auv_count, models in models_dict.items():
    for model in models.values():
        model.fit(scaled_inputs[auv_count], owa_output[owa_input[:, 0] == auv_count])

# --------------------- PREDICTION FUNCTION WITH NORMALIZATION ---------------------
def normalize_weights(weights):
    total = np.sum(weights)
    if total == 0:
        return np.array([10/3, 10/3, 10/3])
    return (weights / total) * 10

def predict_for_testset(test_df, models_dict, scalers_dict):
    rows = []
    for auv in sorted(test_df['auv_count'].unique()):
        test_subset = test_df[test_df['auv_count'] == auv].reset_index(drop=True)
        input_scaled = scalers_dict[auv].transform(test_subset[['auv_count', 'area']].values)
        true_output = test_subset[['w1', 'w2', 'w3', 'utility']].values

        for name, model in models_dict[auv].items():
            preds = model.predict(input_scaled)
            for i in range(len(test_subset)):
                raw_weights = preds[i][:3]
                norm_weights = normalize_weights(raw_weights)
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

comparison_df = predict_for_testset(test_df, models_dict, scalers_dict)
comparison_df.to_csv("results/owa_model_predictions_on_test.csv", index=False)
print("\n📁 Predictions on the test set saved to 'results/owa_model_predictions_on_test.csv'")

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

test_metrics_df = pd.DataFrame(test_metrics_summary).sort_values(["AUV Count", "MAE"])
test_metrics_df.to_csv("results/owa_model_test_metrics.csv", index=False)
print("\n📁 Test metrics saved to 'results/owa_model_test_metrics.csv'")

# --------------------- PLOTS ---------------------
df_long = owa_df.melt(id_vars=["area", "auv_count", "utility"], 
                  value_vars=["w1", "w2", "w3"],
                  var_name="weight_type",
                  value_name="weight_value")

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_long, x="area", y="weight_value", hue="weight_type", marker='o')
plt.title("w1, w2, w3 vs Exploration Area")
plt.xlabel("Exploration Area")
plt.ylabel("Weight Value")
plt.grid(True)
plt.legend(title="Weight")
plt.tight_layout()
plt.show()

# Add a combined column for AUV and model
test_metrics_df["Group"] = test_metrics_df["AUV Count"].astype(str) + " AUV - " + test_metrics_df["Model"]

# Sort by AUV Count and MAE
sorted_df = test_metrics_df.sort_values(["AUV Count", "MAE"])
custom_order = sorted_df["Group"].values

# MAE bar plot
plt.figure(figsize=(16, 6))
sns.barplot(
    data=sorted_df,
    x="Group",
    y="MAE",
    palette="viridis",
    order=custom_order
)
plt.title("MAE Comparison by Model Within Each AUV Group")
plt.ylabel("Mean Absolute Error (MAE)")
plt.xlabel("Model by AUV Group")
plt.xticks(rotation=45, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# RMSE bar plot
plt.figure(figsize=(16, 6))
sns.barplot(
    data=sorted_df,
    x="Group",
    y="RMSE",
    palette="magma",
    order=custom_order
)
plt.title("RMSE Comparison by Model Within Each AUV Group")
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.xlabel("Model by AUV Group")
plt.xticks(rotation=45, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --------------------- PREDICTION FOR SPECIFIC VALUES ---------------------
auv_count = 6
area = 55000

if auv_count not in models_dict:
    print(f"No trained models found for {auv_count} AUVs.")
else:
    input_data = np.array([[auv_count, area]])
    input_scaled = scalers_dict[auv_count].transform(input_data)

    print(f"\n📍 Prediction for {auv_count} AUVs and area = {area}:")
    for model_name, model in models_dict[auv_count].items():
        prediction = model.predict(input_scaled)[0]
        norm_weights = normalize_weights(prediction[:3])
        utility = prediction[3] if len(prediction) > 3 else np.nan

        w1, w2, w3 = norm_weights
        print(f"\n🔹 {model_name}:")
        print(f"    w1 = {w1:.3f}, w2 = {w2:.3f}, w3 = {w3:.3f}, utility = {utility:.3f}")


# --------------------- 3D SVR REGRESSION PLOTS ---------------------
X = owa_df[['auv_count', 'area']].values
y = owa_df[['w1', 'w2', 'w3']].values

auv_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
area_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
auv_grid, area_grid = np.meshgrid(auv_range, area_range)
X_grid = np.c_[auv_grid.ravel(), area_grid.ravel()]

fig = plt.figure(figsize=(18, 5))
for i, weight_name in enumerate(['w1', 'w2', 'w3']):
    model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=7, epsilon=1.2, gamma=0.1))
    model.fit(X, y[:, i])
    y_pred_grid = model.predict(X_grid).reshape(auv_grid.shape)

    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    ax.plot_surface(auv_grid, area_grid, y_pred_grid, cmap='viridis', alpha=0.7)
    ax.scatter(X[:, 0], X[:, 1], y[:, i], c='red', s=20)
    ax.set_xlabel('Number of AUVs')
    ax.set_ylabel('Exploration Area Surface [m^2]')
    ax.set_zlabel(weight_name)
    ax.set_title(f'SVM Regression ({weight_name})')

plt.tight_layout()
plt.show()

# 55000
# 3: w1 = 6.224, w2 = 2.832, w3 = 0.944
# 4: w1 = 5.917, w2 = 3.065, w3 = 1.018
# 5:  w1 = 6.666, w2 = 2.387, w3 = 0.947
# 6: w1 = 5.992, w2 = 2.972, w3 = 1.037
