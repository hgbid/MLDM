import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Reading the CSV file with error handling
try:
    cleaned_data = pd.read_csv('../GovData/cleaned_data_gov.csv', encoding='utf-8')
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    raise

# Drop rows with NaN values
cleaned_data = cleaned_data.dropna()

# Define the target variable and features
y = cleaned_data['price_per_sqm']
X = cleaned_data.drop(columns=['price', 'price_per_sqm'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the linear regression model
regr = LinearRegression()
regr.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = regr.predict(X_test_scaled)

# Evaluate the model
mean_error = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
n = X_test.shape[0]
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Output the evaluation metrics and coefficients
print(f"Root Mean Squared Error: {mean_error}")
print(f"R^2 score of the linear regression model: {r2}")
print(f"Adjusted R^2 score of the linear regression model: {adj_r2}")
print(f"Coefficients: {regr.coef_}")

# Optional: visualize the results
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Observed vs Predicted Prices per sqm')
plt.show()
