import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

matplotlib.use('tkagg')

try:
    cleaned_data = pd.read_csv('../outliers_cleaned_data.csv', encoding='utf-8')

except Exception as e:
    print(f"Error reading the CSV file: {e}")
    raise


# ############   DECISION TREE   ############



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_

indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print(f"{f + 1}. feature {X_train.columns[indices[f]]} ({importances[indices[f]]})")

# Plot the feature importances of the forest
plt.figure()
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

# Initialize the DecisionTreeRegressor with selected features
reg = DecisionTreeRegressor(random_state=42)
reg.fit(X_train_selected, y_train)

# Make predictions with the selected features
y_pred = reg.predict(X_test_selected)

# Calculate the RMSLE
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

rmsle_value = rmsle(y_test, y_pred)

print(f"Root Mean Squared Logarithmic Error with selected features: {rmsle_value}")

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# Define RMSLE scorer
rmsle_scorer = make_scorer(mean_squared_log_error, greater_is_better=False)

# Perform cross-validation
scores = cross_val_score(reg, X_train_selected, y_train, cv=5, scoring=rmsle_scorer)
rmsle_scores = np.sqrt(-scores)  # Convert to positive RMSLE scores

print(f"Cross-validated RMSLE: {rmsle_scores.mean()} ± {rmsle_scores.std()}")

