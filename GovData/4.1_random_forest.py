# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_log_error, make_scorer
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestRegressor
# import optuna
#
# # Load training data
# try:
#     train_data = pd.read_csv('./clean_gov_dataset.csv', encoding='utf-8')
# except Exception as e:
#     print(f"Error reading the training CSV file: {e}")
#     raise
#
# # Load testing data
# try:
#     test_data = pd.read_csv('../Yad2/clean_yad2_dataset.csv', encoding='utf-8')
# except Exception as e:
#     print(f"Error reading the testing CSV file: {e}")
#     raise
#
# # Drop rows with NaN values
# train_data = train_data.dropna()
# test_data = test_data.dropna()
#
# # Prepare training features and target
# X_train = train_data.drop(columns=['price', 'price_per_sqm'])
# y_train = train_data['price']
# print(f'len(X_train) {len(X_train)}')
#
# # Prepare testing features and target
# X_test = test_data.drop(columns=['price', 'price_per_sqm'])
# y_test = test_data['price_per_sqm']
# print(f'len(X_test) {len(X_test)}')
#
# # Ensure that the training and test sets have the same features
# print(X_train.columns, X_test.columns)
# common_features = X_train.columns.intersection(X_test.columns)
# print(common_features)
# X_train = X_train[common_features]
# X_test = X_test[common_features]
#
# # Initialize and fit the RandomForestRegressor
# rf = RandomForestRegressor(random_state=42)
# rf.fit(X_train, y_train)
#
# importances = rf.feature_importances_
# indices = np.argsort(importances)[::-1]
#
# # Print the feature ranking
# print("Feature ranking:")
# for f in range(X_train.shape[1]):
#     print(f"{f + 1}. feature {X_train.columns[indices[f]]} ({importances[indices[f]]})")
#
# # Plot the feature importances
# plt.figure(figsize=(12, 6))
# plt.title("Feature importances")
# plt.bar(range(X_train.shape[1]), importances[indices], align="center")
# plt.xticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices], rotation=90)
# plt.xlim([-1, X_train.shape[1]])
# plt.show()
#
# # Select top N features (e.g., top 10 features)
# N = 10
# top_features = common_features[indices[:N]]
#
# # Reduce the dataset to the top N features
# X_train_selected = X_train[top_features]
# X_test_selected = X_test[top_features]
#
# # Define the objective function for Optuna
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
#     scores = cross_val_score(rf_tuned, X_train_selected, y_train, cv=3, scoring='neg_mean_squared_log_error')
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
# rf_best.fit(X_train_selected, y_train)
# y_pred = rf_best.predict(X_test_selected)
#
# # Add the predictions to the original test dataset
# test_data['gov_prediction'] = y_pred
# test_data.to_csv('yad2_with_predictions.csv', index=False)
#
# print("Predictions added to the dataset and saved to 'yad2_with_predictions.csv'")
#
# # Calculate the RMSLE
# rmsle_value = np.sqrt(mean_squared_log_error(y_test, y_pred))
# print(f"Root Mean Squared Logarithmic Error with tuned hyperparameters: {rmsle_value}")
# rmsle_scorer = make_scorer(mean_squared_log_error, greater_is_better=False)
# scores = cross_val_score(rf_best, X_train_selected, y_train, cv=5, scoring=rmsle_scorer)
# rmsle_scores = np.sqrt(-scores)  # Convert to positive RMSLE scores
# print(f"Cross-validated RMSLE: {rmsle_scores.mean()} ± {rmsle_scores.std()}")
#
# # Plot observed vs predicted prices per sqm
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.3)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
# plt.xlabel('Observed')
# plt.ylabel('Predicted')
# plt.title('Observed vs Predicted Prices per sqm')
# plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Load training data
try:
    train_data = pd.read_csv('./clean_gov_dataset.csv', encoding='utf-8')
except Exception as e:
    print(f"Error reading the training CSV file: {e}")
    raise

# Load testing data
try:
    test_data = pd.read_csv('../Yad2/clean_yad2_dataset.csv', encoding='utf-8')
except Exception as e:
    print(f"Error reading the testing CSV file: {e}")
    raise

# Drop rows with NaN values
train_data = train_data.dropna()
test_data = test_data.dropna()

# Prepare training features and target
X_train = train_data.drop(columns=['price', 'price_per_sqm'])
y_train = train_data['price_per_sqm']
print(f'len(X_train) {len(X_train)}')
# Prepare testing features and target
X_test = test_data.drop(columns=['price', 'price_per_sqm'])
y_test = test_data['price_per_sqm']
print(f'len(X_test) {len(X_test)}')

# Ensure that the training and test sets have the same features
print(X_train.columns, X_test.columns)
common_features = X_train.columns.intersection(X_test.columns)
print(common_features)
X_train = X_train[common_features]
X_test = X_test[common_features]

# Initialize and fit the RandomForestRegressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
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
top_features = common_features[indices[:N]]

# Reduce the dataset to the top N features
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# Initialize the RandomForestRegressor with selected features
rf_selected = RandomForestRegressor(random_state=42)
rf_selected.fit(X_train_selected, y_train)
y_pred = rf_selected.predict(X_test_selected)

# Add the predictions to the original test dataset
test_data['gov_prediction'] = y_pred
test_data.to_csv('yad2_with_predictions.csv', index=False)

print("Predictions added to the dataset and saved to 'yad2_with_predictions.csv'")

# Calculate the RMSLE
rmsle_value = np.sqrt(mean_squared_log_error(y_test, y_pred))
print(f"Root Mean Squared Logarithmic Error with selected features: {rmsle_value}")
rmsle_scorer = make_scorer(mean_squared_log_error, greater_is_better=False)
scores = cross_val_score(rf_selected, X_train_selected, y_train, cv=5, scoring=rmsle_scorer)
rmsle_scores = np.sqrt(-scores)  # Convert to positive RMSLE scores
print(f"Cross-validated RMSLE: {rmsle_scores.mean()} ± {rmsle_scores.std()}")

# Plot observed vs predicted prices per sqm
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Observed vs Predicted Prices per sqm')
plt.show()
