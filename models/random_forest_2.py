import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Load training data
try:
    train_data = pd.read_csv('../GovData/outliers_cleaned_data.csv', encoding='utf-8')
except Exception as e:
    print(f"Error reading the training CSV file: {e}")
    raise

# Load testing data
try:
    test_data = pd.read_csv('../outliers_cleaned_data.csv', encoding='utf-8')
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
print(X_train.columns, X_test.columns)
common_features = X_train.columns.intersection(X_test.columns)
print(common_features)
X_train = X_train[common_features]
X_test = X_test[common_features]

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
test_data.to_csv('outliers_cleaned_data_with_predictions.csv', index=False)

print("Predictions added to the dataset and saved to 'outliers_cleaned_data_with_predictions.csv'")

# Calculate the RMSLE
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

rmsle_value = rmsle(y_test, y_pred)
print(f"Root Mean Squared Logarithmic Error with selected features: {rmsle_value}")

# Perform cross-validation
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
