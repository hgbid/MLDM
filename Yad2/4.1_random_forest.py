import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, make_scorer, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
# Load data
try:
    cleaned_data = pd.read_csv('./yad2_with_predictions.csv', encoding='utf-8')
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    raise

# Drop rows with NaN values
cleaned_data = cleaned_data.dropna()

# Define features and target
X = cleaned_data[['square_meters', 'rooms', 'floor', 'date', 'latitude', 'longitude',
       'new_building', 'is_kottage', 'has_yard',
       'university_distance', 'central_station_distance',
       'sami_shamoon_distance', 'soroka_distance', 'north_train_distance',
       'center_train_distance', 'grand_mall_distance', 'old_city_distance', 'gov_prediction']]
y = cleaned_data['price_per_sqm']

# Calculate VIF for each feature
X_with_constant = sm.add_constant(X)  # Adding a constant to the model
vif = pd.DataFrame()
vif['Feature'] = X.columns
vif['VIF'] = [variance_inflation_factor(X_with_constant.values, i+1) for i in range(X_with_constant.shape[1] - 1)]

print(vif)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the RandomForestRegressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. feature {X_train.columns[indices[f]]} ({importances[indices[f]]})")

# Plot the feature importances
plt.figure(figsize=(12, 6))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# Select top N features (e.g., top 10 features)
N = 10
top_features = indices[:N]

# Reduce the dataset to the top N features
X_train_selected = X_train.iloc[:, top_features]
X_test_selected = X_test.iloc[:, top_features]

# Initialize the RandomForestRegressor with selected features
rf_selected = RandomForestRegressor(random_state=42)
rf_selected.fit(X_train_selected, y_train)

# Make predictions with the selected features
y_pred = rf_selected.predict(X_test_selected)

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

rmsle_value = rmsle(y_test, y_pred)
print(f"Root Mean Squared Logarithmic Error with selected features: {rmsle_value}")

# Perform cross-validation
rmsle_scorer = make_scorer(mean_squared_log_error, greater_is_better=False)
scores = cross_val_score(rf_selected, X_train_selected, y_train, cv=5, scoring=rmsle_scorer)
rmsle_scores = np.sqrt(-scores)  # Convert to positive RMSLE scores
print(f"Cross-validated RMSLE: {rmsle_scores.mean()} ± {rmsle_scores.std()}")

mse_value = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse_value}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Observed vs Predicted Prices per sqm')
plt.show()
# import optuna
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_log_error, make_scorer, mean_squared_error
# from sklearn.feature_selection import RFE
#
# # Load data
# try:
#     cleaned_data = pd.read_csv('./yad2_with_predictions.csv', encoding='utf-8')
# except Exception as e:
#     print(f"Error reading the CSV file: {e}")
#     raise
#
# # Drop rows with NaN values
# cleaned_data = cleaned_data.dropna()
# # Define features and target variable
# X = cleaned_data[['square_meters', 'rooms', 'floor', 'is_ground', 'new_building',
#        'is_kottage',  'has_yard', 'university_distance',
#        'central_station_distance', 'sami_shamoon_distance', 'soroka_distance',
#        'north_train_distance', 'center_train_distance', 'grand_mall_distance',
#        'old_city_distance','date', 'gov_prediction']]
# y = cleaned_data['price']
#
# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# def objective(trial):
#     param = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'max_depth': trial.suggest_int('max_depth', 10, 50),
#         'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
#         'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
#         'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
#         'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
#     }
#
#     rf_tuned = RandomForestRegressor(random_state=42, **param)
#     scores = cross_val_score(rf_tuned, X_train, y_train, cv=3, scoring='neg_mean_squared_log_error')
#     rmsle_scores = np.sqrt(-scores)
#     return rmsle_scores.mean()
#
# # Optimize hyperparameters using Optuna
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)
#
# # Get the best parameters and train the model
# best_params = study.best_params
# print("Best parameters found by Optuna:", best_params)
#
# rf_best = RandomForestRegressor(random_state=42, **best_params)
# rf_best.fit(X_train, y_train)
# y_pred = rf_best.predict(X_test)
#
# # # Get the selected features
# # selected_features = X_train.columns[rf_best.support_]
# #
# # print("Selected features:", selected_features)
# #
# # # Reduce the dataset to the selected features
# # X_train_selected = X_train[selected_features]
# # X_test_selected = X_test[selected_features]
# #
# # # Initialize the RandomForestRegressor with selected features
# # rf_selected = RandomForestRegressor(random_state=42)
# # rf_selected.fit(X_train_selected, y_train)
# #
# # # Make predictions with the selected features
# # y_pred = rf_selected.predict(X_test_selected)
#
# # Define RMSLE function
# def rmsle(y_true, y_pred):
#     return np.sqrt(mean_squared_log_error(y_true, y_pred))
#
# rmsle_value = rmsle(y_test, y_pred)
# print(f"Root Mean Squared Logarithmic Error with selected features: {rmsle_value}")
#
# # Perform cross-validation
# rmsle_scorer = make_scorer(mean_squared_log_error, greater_is_better=False)
# scores = cross_val_score(rf_best, X_train, y_train, cv=5, scoring=rmsle_scorer)
# rmsle_scores = np.sqrt(-scores)  # Convert to positive RMSLE scores
# print(f"Cross-validated RMSLE: {rmsle_scores.mean()} ± {rmsle_scores.std()}")
#
# # Calculate Mean Squared Error
# mse_value = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse_value}")
#
# # Plot observed vs predicted prices per sqm
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.3)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
# plt.xlabel('Observed')
# plt.ylabel('Predicted')
# plt.title('Observed vs Predicted Prices per sqm')
# plt.show()
#
# # Plot feature importances
# importances = rf_selected.feature_importances_
# indices = np.argsort(importances)[::-1]
# plt.figure(figsize=(12, 6))
# plt.title("Feature importances")
# plt.bar(range(X_train_selected.shape[1]), importances[indices], align="center")
# plt.xticks(range(X_train_selected.shape[1]), [X_train_selected.columns[i] for i in indices], rotation=90)
# plt.xlim([-1, X_train_selected.shape[1]])
# plt.show()
