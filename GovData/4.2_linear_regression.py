import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

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

# Initialize and fit the Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Make predictions
y_pred_lr = model_lr.predict(X_test)

# Calculate metrics for evaluation
mse_value_lr = mean_squared_error(y_test, y_pred_lr)
r2_value_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression Results:")
print(f"Mean Squared Error (MSE): {mse_value_lr}")
print(f"R-squared (R2): {r2_value_lr}")

# Perform cross-validation
scores_lr = cross_val_score(model_lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
mse_scores_lr = -scores_lr  # Convert to positive MSE scores
print(f"Cross-validated MSE: {mse_scores_lr.mean()} ± {mse_scores_lr.std()}")

# Plot observed vs predicted prices per sqm for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Observed vs Predicted Prices per sqm (Linear Regression)')
plt.show()


#########################################
# Initialize and fit the Polynomial Regression model
degree = 2  # Example degree of polynomial (adjust as needed)
model_pr = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model_pr.fit(X_train, y_train)

# Make predictions
y_pred_pr = model_pr.predict(X_test)

# Calculate metrics for evaluation
mse_value_pr = mean_squared_error(y_test, y_pred_pr)
r2_value_pr = r2_score(y_test, y_pred_pr)

print(f"Polynomial Regression (Degree {degree}) Results:")
print(f"Mean Squared Error (MSE): {mse_value_pr}")
print(f"R-squared (R2): {r2_value_pr}")

# Perform cross-validation
scores_pr = cross_val_score(model_pr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
mse_scores_pr = -scores_pr  # Convert to positive MSE scores
print(f"Cross-validated MSE: {mse_scores_pr.mean()} ± {mse_scores_pr.std()}")

# Plot observed vs predicted prices per sqm for Polynomial Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_pr, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title(f'Observed vs Predicted Prices per sqm (Polynomial Regression - Degree {degree})')
plt.show()
