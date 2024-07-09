import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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

# Prepare testing features and target
X_test = test_data.drop(columns=['price', 'price_per_sqm'])
y_test = test_data['price_per_sqm']

# Ensure that the training and test sets have the same features
common_features = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_features]
X_test = X_test[common_features]

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define SVR model
svr = SVR()

# Define parameter grid
param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best parameters and best score
print("Best parameters found:", grid_search.best_params_)
print("Best negative MSE found:", grid_search.best_score_)

# Evaluate on test set with best model
best_svr = grid_search.best_estimator_
y_pred_svr = best_svr.predict(X_test_scaled)

# Calculate metrics
mse_value_svr = mean_squared_error(y_test, y_pred_svr)
r2_value_svr = r2_score(y_test, y_pred_svr)

print("SVR Results after GridSearchCV Optimization:")
print(f"Mean Squared Error (MSE): {mse_value_svr}")
print(f"R-squared (R2): {r2_value_svr}")

# Plot observed vs predicted prices per sqm for SVR
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_svr, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Observed vs Predicted Prices per sqm (SVR after GridSearchCV)')
plt.show()
