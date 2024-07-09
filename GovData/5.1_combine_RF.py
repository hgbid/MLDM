import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_log_error, make_scorer, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load datasets
try:
    gov_data = pd.read_csv('./clean_gov_dataset.csv', encoding='utf-8')
    gov_data['source'] = 1  # Adding source feature for gov dataset
except Exception as e:
    print(f"Error reading the gov CSV file: {e}")
    raise

try:
    yad2_data = pd.read_csv('../Yad2/clean_yad2_dataset.csv', encoding='utf-8')
    yad2_data['source'] = 0  # Adding source feature for yad2 dataset
except Exception as e:
    print(f"Error reading the yad2 CSV file: {e}")
    raise

# Combine datasets
combined_data = pd.concat([gov_data, yad2_data], ignore_index=True)

# Drop rows with NaN values
combined_data = combined_data.dropna()

# Prepare features and target
X = combined_data.drop(columns=['price', 'price_per_sqm'])
y = combined_data['price_per_sqm']

# Ensure that the combined dataset has only common features
common_features = gov_data.columns.intersection(yad2_data.columns).difference(['price', 'price_per_sqm'])
X = X[common_features]

# Split the combined dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
top_features = X_train.columns[indices[:N]]

# Reduce the dataset to the top N features
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# Initialize the RandomForestRegressor with selected features
rf_selected = RandomForestRegressor(random_state=42)
rf_selected.fit(X_train_selected, y_train)
y_pred = rf_selected.predict(X_test_selected)

# Add the predictions to the original test dataset
combined_data.loc[X_test.index, 'gov_prediction'] = y_pred

# Save the combined dataset with predictions
combined_data.to_csv('combined_with_predictions.csv', index=False)
print("Predictions added to the combined dataset and saved to 'combined_with_predictions.csv'")

# Calculate the RMSLE
rmsle_value = np.sqrt(mean_squared_log_error(y_test, y_pred))
print(f"Root Mean Squared Logarithmic Error with selected features: {rmsle_value}")

# Cross-validated RMSLE
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
