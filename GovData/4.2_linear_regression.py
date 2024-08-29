import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

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

# Filter out negative values in the target variables
X_train, y_train = X_train[y_train >= 0], y_train[y_train >= 0]
X_test, y_test = X_test[y_test >= 0], y_test[y_test >= 0]

#########################################
# Linear Regression
#########################################

# Initialize and fit the Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Make predictions
y_pred_lr = model_lr.predict(X_test)

# Calculate metrics for evaluation
mse_value_lr = mean_squared_error(y_test, y_pred_lr)
r2_value_lr = r2_score(y_test, y_pred_lr)
rmsle_value_lr = np.sqrt(mean_squared_log_error(y_test, y_pred_lr))

# Perform cross-validation for RMSLE
rmsle_scorer = make_scorer(mean_squared_log_error, greater_is_better=False)
cross_val_rmsle_lr = cross_val_score(model_lr, X_train, y_train, cv=5, scoring=rmsle_scorer)
cross_val_rmsle_lr = np.sqrt(-cross_val_rmsle_lr)  # Convert to positive RMSLE scores

print("Linear Regression Results:")
print(f"RMSLE: {rmsle_value_lr}")
print(f"Cross-validated RMSLE: {cross_val_rmsle_lr.mean()} ± {cross_val_rmsle_lr.std()}")
print(f"Mean Squared Error (MSE): {mse_value_lr}")
print()

# Plot observed vs predicted prices per sqm for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Observed vs Predicted Prices per sqm (Linear Regression)')
plt.show()

#########################################
# Polynomial Regression
#########################################

# Initialize and fit the Polynomial Regression model
degree = 2  # Example degree of polynomial (adjust as needed)
model_pr = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model_pr.fit(X_train, y_train)

# Make predictions
y_pred_pr = model_pr.predict(X_test)
y_pred_pr = np.maximum(y_pred_pr, 0)

# Calculate metrics for evaluation
mse_value_pr = mean_squared_error(y_test, y_pred_pr)
r2_value_pr = r2_score(y_test, y_pred_pr)
rmsle_value_pr = np.sqrt(mean_squared_log_error(y_test, y_pred_pr))

# Perform cross-validation for RMSLE
cross_val_rmsle_pr = cross_val_score(model_pr, X_train, y_train, cv=5, scoring=rmsle_scorer)
cross_val_rmsle_pr = np.sqrt(-cross_val_rmsle_pr)  # Convert to positive RMSLE scores

print(f"Polynomial Regression (Degree {degree}) Results:")
print(f"RMSLE: {rmsle_value_pr}")
print(f"Cross-validated RMSLE: {cross_val_rmsle_pr.mean()} ± {cross_val_rmsle_pr.std()}")
print(f"Mean Squared Error (MSE): {mse_value_pr}")
print()

# Plot observed vs predicted prices per sqm for Polynomial Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_pr, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title(f'Observed vs Predicted Prices per sqm (Polynomial Regression - Degree {degree})')
plt.show()
