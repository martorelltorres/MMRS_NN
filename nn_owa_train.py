import numpy as np
import pandas as pd
from itertools import product
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn import neighbors
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Load OWA training data
train_df = pd.read_csv('/home/antoni/MMRS_ws/src/MMRS_stack/MMRS_NN/data/utility_function/all_owa_data.csv')
owa_input_data = train_df[['auv_count', 'area', 'w1', 'w2', 'w3']].values
owa_output_data = train_df[['utility']].values

owa_dtree = DecisionTreeRegressor(max_depth=5)
owa_dtree.fit(owa_input_data, owa_output_data)

randf = RandomForestRegressor(n_estimators=1000)
randf.fit(owa_input_data, owa_output_data)

svr = svm.SVR( kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(owa_input_data, owa_output_data)

knn = neighbors.KNeighborsRegressor(n_neighbors=5)  
knn.fit(owa_input_data, owa_output_data)         

poly_model = make_pipeline(PolynomialFeatures(degree=4), LinearRegression())
poly_model.fit(owa_input_data, owa_output_data)

poly_ridge = make_pipeline(PolynomialFeatures(degree=4), Ridge(alpha=1.0))
poly_ridge.fit(owa_input_data, owa_output_data)

poly_lasso = make_pipeline(PolynomialFeatures(degree=4), Lasso(alpha=0.1, max_iter=10000))
poly_lasso.fit(owa_input_data, owa_output_data)

# Load ARTM training data
train_df = pd.read_csv('/home/antoni/MMRS_ws/src/MMRS_stack/MMRS_NN/data/utility_function/all_artm_data.csv')
owa_input_data = train_df[['auv_count', 'area', 'a', 'b']].values
owa_output_data = train_df[['utility']].values


# Function to find the best w1, w2, w3 for a given auv_count and area
def find_optimal_owa_weights(auv_count, area, regressor):
    owa_combinations = [(10, 0, 0), (9, 1, 0), (8, 2, 0), (8, 2, 1), (8, 1, 1),
                           (7, 3, 0), (7, 2, 1), (6, 4, 0), (6, 2, 2), (6, 3, 1),
                           (5, 5, 0), (5, 4, 1), (5, 3, 2), (4, 4, 2), (4, 3, 3)]
    test_df = pd.DataFrame(owa_combinations, columns=['w1', 'w2', 'w3'])
    test_df['auv_count'] = auv_count
    test_df['area'] = area

    predicted_utilities = regressor.predict(test_df[['auv_count', 'area', 'w1', 'w2', 'w3']].values)
    
    # Find the best combination that maximizes utility
    best_idx = np.argmax(predicted_utilities)
    best_w1, best_w2, best_w3 = owa_combinations[best_idx]
    best_utility = predicted_utilities[best_idx].item()

    # print(predicted_utilities)
    return best_w1, best_w2, best_w3, best_utility

# Function to find the best w1, w2, w3 for a given auv_count and area
def find_optimal_artm_weights(auv_count, area, regressor):
    artm_combinations = [(10, 0), (9, 1), (8, 2), (7, 3), (6, 4), (5, 5),
                           (4, 6), (3, 7), (2, 8), (1, 9), (0, 10)]
    test_df = pd.DataFrame(artm_combinations, columns=['a', 'b'])
    test_df['auv_count'] = auv_count
    test_df['area'] = area

    predicted_utilities = regressor.predict(test_df[['auv_count', 'area', 'w1', 'w2', 'w3']].values)
    
    # Find the best combination that maximizes utility
    best_idx = np.argmax(predicted_utilities)
    best_w1, best_w2, best_w3 = artm_combinations[best_idx]
    best_utility = predicted_utilities[best_idx].item()

    # print(predicted_utilities)
    return best_w1, best_w2, best_w3, best_utility

auv_count = 3
area = 25000
print("-----------------------------------------------------")
best_w1, best_w2, best_w3, best_utility = find_optimal_owa_weights(auv_count, area, dtree)
print(f"Optimal weights for auv_count={auv_count}, area={area} using DECISSION TREES:")
print(f"w1 = {best_w1}, w2 = {best_w2}, w3 = {best_w3} (Utility = {best_utility:.3f})")
print("-----------------------------------------------------")

best_w1, best_w2, best_w3, best_utility = find_optimal_owa_weights(auv_count, area, randf)
print(f"Optimal weights for auv_count={auv_count}, area={area} using RANDOM FOREST:")
print(f"w1 = {best_w1}, w2 = {best_w2}, w3 = {best_w3} (Utility = {best_utility:.3f})")
print("-----------------------------------------------------")

best_w1, best_w2, best_w3, best_utility = find_optimal_owa_weights(auv_count, area, svr)
print(f"Optimal weights for auv_count={auv_count}, area={area} using SUPPORT VECTOR REGRESSION:")
print(f"w1 = {best_w1}, w2 = {best_w2}, w3 = {best_w3} (Utility = {best_utility:.3f})")
print("-----------------------------------------------------")

best_w1, best_w2, best_w3, best_utility = find_optimal_owa_weights(auv_count, area, knn)
print(f"Optimal weights for auv_count={auv_count}, area={area} using NEAREST NEIGHBORS:")
print(f"w1 = {best_w1}, w2 = {best_w2}, w3 = {best_w3} (Utility = {best_utility:.3f})")
print("-----------------------------------------------------")

best_w1, best_w2, best_w3, best_utility = find_optimal_owa_weights(auv_count, area, poly_model)
print(f"Optimal weights for auv_count={auv_count}, area={area} using POLYNOMIAL REGRESSION:")
print(f"w1 = {best_w1}, w2 = {best_w2}, w3 = {best_w3} (Utility = {best_utility:.3f})")
print("-----------------------------------------------------")

best_w1, best_w2, best_w3, best_utility = find_optimal_owa_weights(auv_count, area, poly_ridge)
print(f"Optimal weights for auv_count={auv_count}, area={area} using POLYNOMIAL RIDGE:")
print(f"w1 = {best_w1}, w2 = {best_w2}, w3 = {best_w3} (Utility = {best_utility:.3f})")
print("-----------------------------------------------------")

best_w1, best_w2, best_w3, best_utility = find_optimal_owa_weights(auv_count, area, poly_lasso)
print(f"Optimal weights for auv_count={auv_count}, area={area} using POLYNOMIAL LASSO:")
print(f"w1 = {best_w1}, w2 = {best_w2}, w3 = {best_w3} (Utility = {best_utility:.3f})")

