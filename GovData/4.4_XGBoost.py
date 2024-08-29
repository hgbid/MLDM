import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, make_scorer
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load training data
try:
    train_data = pd.read_csv('../Yad2/clean_gov_dataset.csv', encoding='utf-8')
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
y_train = train_data['price']

# Prepare testing features and target
X_test = test_data.drop(columns=['price', 'price_per_sqm'])
y_test = test_data['price']

# Ensure that the training and test sets have the same features
common_features = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_features]
X_test = X_test[common_features]

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define XGBoost regressor
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best parameters and best score
print("Best parameters found:", grid_search.best_params_)
print("Best negative MSE found:", grid_search.best_score_)

# Evaluate on test set with best model
best_xgb = grid_search.best_estimator_
y_pred_xgb = best_xgb.predict(X_test_scaled)

# Calculate metrics
mse_value_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_value_xgb = r2_score(y_test, y_pred_xgb)

print("XGBoost Results after GridSearchCV Optimization:")
print(f"Mean Squared Error (MSE): {mse_value_xgb}")
print(f"R-squared (R2): {r2_value_xgb}")

# Calculate the RMSLE
rmsle_value_xgb = np.sqrt(mean_squared_log_error(y_test, y_pred_xgb))
print(f"Root Mean Squared Logarithmic Error: {rmsle_value_xgb}")

# Cross-validated RMSLE
rmsle_scorer = make_scorer(mean_squared_log_error, greater_is_better=False)
scores = cross_val_score(best_xgb, X_train_scaled, y_train, cv=5, scoring=rmsle_scorer)
rmsle_scores = np.sqrt(-scores)  # Convert to positive RMSLE scores

print(f"Cross-validated RMSLE: {rmsle_scores.mean()} ± {rmsle_scores.std()}")

# Plot observed vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Observed vs Predicted Prices')
plt.show()

# Add the predictions to the original test dataset
test_data['gov_prediction'] = y_pred_xgb
test_data.to_csv('yad2_with_predictions.csv', index=False)
print("Predictions added to the dataset and saved to 'yad2_with_predictions.csv'")
