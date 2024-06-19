import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

matplotlib.use('tkagg')

try:
    cleaned_data = pd.read_csv('outliers_cleaned_data.csv', encoding='utf-8')

except Exception as e:
    print(f"Error reading the CSV file: {e}")
    raise

X = cleaned_data.drop(columns=['price', 'price_per_sqm'])
y = cleaned_data['price']

# ############ LINEAR REGRESSION ############
# regr = linear_model.LinearRegression()
# regr.fit(X, y)
#
# y_pred = regr.predict(X)
# print(f"Mean Error {(abs(y-y_pred)).mean()}")
#
# r2 = r2_score(y, y_pred)
# print(f"R^2 score of the linear regression model: {r2}")
#
# n = X.shape[0]
# p = X.shape[1]
# adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
# print(f"Adjusted R^2 score of the linear regression model: {adj_r2}")

print(f"Coefficients: {regr.coef_}")

# ############   DECISION TREE   ############

# bins = pd.interval_range(start=0, end=y.max()+50000, freq=50000)
# # y_binned = pd.cut(y, bins=bins, labels=False, include_lowest=True).astype(int)
# y_binned = y//50000
#
# X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=42)
#
# clf = DecisionTreeClassifier(random_state=42)
# clf.fit(X_train, y_train)
#
# y_pred = clf.predict(X_test)
#
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

########################

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Define the features and target
X = cleaned_data[['rooms', 'floor', 'condition', 'is_promoted', 'square_meters','latitude', 'longitude']]
y = cleaned_data['price']

# Bin the prices into 10k intervals
y_binned = y // 10000

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the target to categorical
y_train = tf.keras.utils.to_categorical(y_train, num_classes=int(y_binned.max() + 1))
y_test = tf.keras.utils.to_categorical(y_test, num_classes=int(y_binned.max() + 1))

# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(int(y_binned.max() + 1), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print(f"Accuracy: {accuracy_score(y_test_classes, y_pred_classes)}")
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes))

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.show()
