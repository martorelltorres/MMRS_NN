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
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import os

REPO = os.path.dirname(os.path.abspath(__file__))

# --------------------- LOAD TRAINING DATA ---------------------
paths = [
    os.path.join(REPO, 'weights', '3AUV_weights.csv'),
    os.path.join(REPO, 'weights', '4AUV_weights.csv'),
    os.path.join(REPO, 'weights', '5AUV_weights.csv'),
    os.path.join(REPO, 'weights', '6AUV_weights.csv'),
]

owa_df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
owa_input = owa_df[['auv_count', 'area']].values
owa_output = owa_df[['w1', 'w2', 'w3', 'utility']].values

test_path = os.path.join(REPO, 'weights', 'optimal_test_weights.csv')  
test_df = pd.read_csv(test_path)

# --------------------- SCALING ---------------------
# One scaler over the whole training set. Fitting a separate scaler per fleet size left
# auv_count with zero variance inside each subset, so StandardScaler collapsed it to a
# constant and every model effectively saw a single feature with six samples.
scaler = StandardScaler()
scaled_input = scaler.fit_transform(owa_input)

# --------------------- HYPERPARAMETER SELECTION ---------------------
# Set to False to reproduce the published results with the parameters of Table 1.
TUNE_HYPERPARAMETERS = True

# Published configuration, kept so the previous numbers stay reproducible.
PUBLISHED_PARAMS = {
    "Decision Tree": {"max_depth": 5, "min_samples_leaf": 3},
    "Random Forest": {"n_estimators": 1000},
    "SVR": {"kernel": "rbf", "C": 7, "epsilon": 1.2, "gamma": 0.1},
    "Lasso": {"alpha": 0.3},
}

# Every model that is compared gets its own search, not just the SVR: tuning one technique
# and leaving the rest at hand-picked values would make the comparison of Figures 9 and 10
# meaningless.
PARAM_GRIDS = {
    "Decision Tree": {"estimator__max_depth": [2, 3, 5, 8, None],
                      "estimator__min_samples_leaf": [1, 2, 3, 5]},
    "Random Forest": {"estimator__n_estimators": [100, 300, 1000],
                      "estimator__max_depth": [2, 3, 5, None]},
    # The kernel is searched rather than assumed: the published configuration fixed the RBF
    # by hand, and the choice turns out to matter more than any of its parameters. The grid is
    # kept coarse on purpose -- with 24 samples a finer one would only fit the validation
    # folds -- but it spans the range where both kernels have their optimum.
    "SVR": {"estimator__kernel": ["rbf", "poly"],
            "estimator__degree": [2, 3],
            "estimator__C": [0.1, 1, 10, 100],
            "estimator__epsilon": [0.01, 0.05, 0.1, 0.3, 1.0],
            "estimator__gamma": ["scale", 0.1, 1.0, 10.0]},
    "Polynomial": {"estimator__polynomialfeatures__degree": [1, 2, 3, 4]},
    "Lasso": {"estimator__alpha": [0.001, 0.01, 0.05, 0.1, 0.3, 1.0]},
}


def base_models():
    """Model templates. Hyperparameters are either searched or taken from PUBLISHED_PARAMS."""
    return {
        "Decision Tree": MultiOutputRegressor(DecisionTreeRegressor(random_state=0)),
        "Random Forest": MultiOutputRegressor(RandomForestRegressor(random_state=0)),
        # max_iter caps the pathological corners of the grid: a polynomial kernel with large
        # C and gamma can leave libsvm iterating without converging, which stalls the whole
        # search. Combinations that hit the cap simply score badly and lose.
        "SVR": MultiOutputRegressor(SVR(kernel='rbf', max_iter=200000)),
        "Polynomial": MultiOutputRegressor(make_pipeline(PolynomialFeatures(), LinearRegression())),
        "Lasso": MultiOutputRegressor(Lasso(max_iter=10000)),
        # Reference point: predicts the mean of the training targets and ignores the inputs
        # entirely. Any model that does not beat it is not using the mission descriptors, so
        # this has to be reported alongside the rest rather than assumed away. It has nothing
        # to tune.
        "Mean baseline": DummyRegressor(strategy='mean'),
    }


def published_models():
    return {
        "Decision Tree": MultiOutputRegressor(DecisionTreeRegressor(**PUBLISHED_PARAMS["Decision Tree"])),
        "Random Forest": MultiOutputRegressor(RandomForestRegressor(**PUBLISHED_PARAMS["Random Forest"])),
        "SVR": MultiOutputRegressor(SVR(**PUBLISHED_PARAMS["SVR"])),
        "Polynomial": MultiOutputRegressor(make_pipeline(PolynomialFeatures(4), LinearRegression())),
        "Lasso": MultiOutputRegressor(Lasso(**PUBLISHED_PARAMS["Lasso"])),
        "Mean baseline": DummyRegressor(strategy='mean'),
    }


def tune(models, X, y):
    """Select hyperparameters by leave-one-out cross-validation on the training set.

    With 24 training samples LOO is the only protocol that leaves anything to validate on:
    a k-fold split would hold out two or three points at a time out of twenty-four. The
    search never sees the test set, so the reported test error stays an honest estimate.
    """
    records = []
    for name, model in models.items():
        grid = PARAM_GRIDS.get(name)
        if grid is None:
            model.fit(X, y)
            records.append({"model": name, "selected": "n/a", "loo_mae": np.nan})
            continue

        search = GridSearchCV(model, grid, cv=LeaveOneOut(),
                              scoring='neg_mean_absolute_error', n_jobs=-1)
        search.fit(X, y)
        models[name] = search.best_estimator_

        selected = {k.replace('estimator__', '').replace('polynomialfeatures__', ''): v
                    for k, v in search.best_params_.items()}
        records.append({"model": name, "selected": str(selected),
                        "loo_mae": -search.best_score_})
        print("  %-15s %-58s LOO MAE %.4f"
              % (name, str(selected), -search.best_score_))

    return models, pd.DataFrame(records)


# A single model per technique, trained on the 24 samples with both descriptors as real
# inputs. Metrics are still reported per fleet size, so the published figures keep their
# structure and stay comparable.
if TUNE_HYPERPARAMETERS:
    print("\nSelecting hyperparameters by leave-one-out cross-validation (%d samples)"
          % len(scaled_input))
    models, search_df = tune(base_models(), scaled_input, owa_output)
    search_df.to_csv("results/hyperparameter_search.csv", index=False)
    print("\nSelected hyperparameters saved to 'results/hyperparameter_search.csv'")
else:
    models = published_models()
    for model in models.values():
        model.fit(scaled_input, owa_output)

models_dict = {auv: models for auv in range(3, 7)}
scalers_dict = {auv: scaler for auv in range(3, 7)}

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


# --------------------- TEST METRICS (PER COMPONENT) ---------------------
test_metrics_summary = []

for auv in sorted(test_df['auv_count'].unique()):
    test_subset = test_df[test_df['auv_count'] == auv]
    test_input = scalers_dict[auv].transform(test_subset[['auv_count', 'area']].values)
    test_output = test_subset[['w1', 'w2', 'w3']].values

    for name, model in models_dict[auv].items():
        preds = model.predict(test_input)
        norm_preds = np.apply_along_axis(normalize_weights, 1, preds[:, :3])

        # Global metrics
        mae_global = mean_absolute_error(test_output, norm_preds)
        rmse_global = mean_squared_error(test_output, norm_preds, squared=False)

        # Per-component metrics
        mae_w1 = mean_absolute_error(test_output[:, 0], norm_preds[:, 0])
        mae_w2 = mean_absolute_error(test_output[:, 1], norm_preds[:, 1])
        mae_w3 = mean_absolute_error(test_output[:, 2], norm_preds[:, 2])

        rmse_w1 = mean_squared_error(test_output[:, 0], norm_preds[:, 0], squared=False)
        rmse_w2 = mean_squared_error(test_output[:, 1], norm_preds[:, 1], squared=False)
        rmse_w3 = mean_squared_error(test_output[:, 2], norm_preds[:, 2], squared=False)

        test_metrics_summary.append({
            "AUV Count": int(auv),
            "Regression Model": name,
            "MAE_global": mae_global,
            "RMSE_global": rmse_global,
            "MAE_w1": mae_w1,
            "MAE_w2": mae_w2,
            "MAE_w3": mae_w3,
            "RMSE_w1": rmse_w1,
            "RMSE_w2": rmse_w2,
            "RMSE_w3": rmse_w3
        })

test_metrics_df = pd.DataFrame(test_metrics_summary)
test_metrics_df.to_csv("results/owa_model_test_metrics_per_component.csv", index=False)
print("\n📁 Per-component test metrics saved.")

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
test_metrics_df["Group"] = test_metrics_df["AUV Count"].astype(str) + " AUV - " + test_metrics_df["Regression Model"]

# Sort by AUV Count and MAE
sorted_df = test_metrics_df.sort_values(["AUV Count", "MAE_global"])
custom_order = sorted_df["Group"].values

# MAE bar plot
plt.figure(figsize=(16, 6))
sns.barplot(
    data=sorted_df,
    x="Group",
    y="MAE_global",
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
    y="RMSE_global",
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

# --------------------- PER-COMPONENT MAE PLOT ---------------------

df_melt = test_metrics_df.melt(
    id_vars=["AUV Count", "Regression Model"],
    value_vars=["MAE_w1", "MAE_w2", "MAE_w3"],
    var_name="Weight",
    value_name="MAE"
)

plt.figure(figsize=(16,6))
ax = sns.barplot(
    data=df_melt,
    x="Regression Model",
    y="MAE",
    hue="Weight",          # importante si tienes w1, w2, w3
    palette="magma",
    errorbar="ci"
)

ax.set_title("Per-Component MAE Comparison", fontsize=18)
ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=16)
ax.set_xlabel("Regression Model", fontsize=16)

ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

plt.xticks(rotation=45)

# Añadir valores encima de cada barra
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=11)

plt.tight_layout()
plt.show()

# --------------------- PER-COMPONENT RMSE PLOT ---------------------

df_melt_rmse = test_metrics_df.melt(
    id_vars=["AUV Count", "Regression Model"],
    value_vars=["RMSE_w1", "RMSE_w2", "RMSE_w3"],
    var_name="Weight",
    value_name="RMSE"
)

plt.figure(figsize=(16,6))
ax = sns.barplot(
    data=df_melt_rmse,
    x="Regression Model",
    y="RMSE",
    hue="Weight",
    palette="magma",
    errorbar="ci"
)

ax.set_title("Per-Component RMSE Comparison", fontsize=18)
ax.set_ylabel("Root Mean Squared Error (RMSE)", fontsize=16)
ax.set_xlabel("Regression Model", fontsize=16)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

plt.xticks(rotation=45)

# ----------------- ADD VALUE LABELS -----------------
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=11)

plt.tight_layout()
plt.show()
# --------------------- PREDICTION FOR SPECIFIC VALUES ---------------------
auv_count = 5
area = 35000

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
for i, weight_name in enumerate([r'$w_1$', r'$w_2$', r'$w_3$']):
    model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=7, epsilon=1.2, gamma=0.1))
    model.fit(X, y[:, i])
    y_pred_grid = model.predict(X_grid).reshape(auv_grid.shape)

    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    ax.plot_surface(auv_grid, area_grid, y_pred_grid, cmap='viridis', alpha=0.7)
    ax.scatter(X[:, 0], X[:, 1], y[:, i], c='red', s=20)
    ax.set_xlabel('Number of AUVs', fontsize=14,labelpad=5)
    ax.set_ylabel('Exploration Area Surface [m²]', fontsize=14,labelpad=15)
    ax.set_zlabel(weight_name,fontsize=14)
    ax.set_title(f'SVM Regression ({weight_name})', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)

plt.tight_layout()
plt.show()

