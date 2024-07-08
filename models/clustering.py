import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data
try:
    cleaned_data = pd.read_csv('../GovData/outliers_cleaned_data.csv', encoding='utf-8')
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    raise

# Drop rows with NaN values
cleaned_data = cleaned_data.dropna()

# Select the specific columns for clustering
X = cleaned_data[['price_per_sqm', 'latitude', 'longitude']]

# Give more weight to price_per_sqm
X['price_per_sqm_weighted'] = X['price_per_sqm'] * 10  # Adjust the weight factor as needed

# Drop the original price_per_sqm column
X.drop(columns=['price_per_sqm'], inplace=True)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try different K values for KMeans
k_values = [3, 4, 5]
best_score = -1
best_kmeans = None

for k in k_values:
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Evaluate clustering performance with silhouette score
    silhouette_avg = silhouette_score(X_scaled, clusters)
    print(f"For K={k}, silhouette score: {silhouette_avg}")

    # Update best score and kmeans model if silhouette score improves
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_kmeans = kmeans

# Assign best clustering results
clusters = best_kmeans.labels_

# Add the cluster labels and original price_per_sqm back to the original dataframe
cleaned_data['cluster'] = clusters
cleaned_data['price_per_sqm'] = X['price_per_sqm_weighted'] / 10  # Rescale to original scale

# Plot the clusters using latitude and longitude as the axes, size as price_per_sqm, and color intensity as price_per_sqm
plt.figure(figsize=(12, 8))

# Define shapes for different clusters
shapes = ['o', 's', '^', 'D', 'x']

for cluster in range(best_kmeans.n_clusters):
    cluster_data = cleaned_data[cleaned_data['cluster'] == cluster]
    plt.scatter(cluster_data['latitude'], cluster_data['longitude'], s=0.01, c=cluster_data['cluster'], cmap='viridis', alpha=0.2)
    plt.scatter(cluster_data['latitude'], cluster_data['longitude'], s=cluster_data['price_per_sqm'] * 0.01,
                c=cluster_data['price_per_sqm'], cmap='plasma', alpha=0.8,
                label=f'Cluster {cluster + 1}')

plt.colorbar(label='Price per sqm')
plt.title('KMeans Clustering with Price per sqm (Bubble Plot)')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend()
plt.show()

# Display cluster centers in the original feature space
cluster_centers = scaler.inverse_transform(best_kmeans.cluster_centers_)
centers_df = pd.DataFrame(cluster_centers, columns=['latitude', 'longitude', 'price_per_sqm'])
print("Cluster centers in original feature space:")
print(centers_df)

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
#
# # Load data
# try:
#     cleaned_data = pd.read_csv('../GovData/outliers_cleaned_data.csv', encoding='utf-8')
# except Exception as e:
#     print(f"Error reading the CSV file: {e}")
#     raise
#
# # Drop rows with NaN values
# cleaned_data = cleaned_data.dropna()
# X = cleaned_data[['price_per_sqm', 'latitude', 'longitude']]
#
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # Apply PCA for dimensionality reduction to 2 dimensions for visualization
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
#
# # Perform KMeans clustering
# kmeans = KMeans(n_clusters=5, random_state=42)
# clusters = kmeans.fit_predict(X_scaled)
#
# # Add the cluster labels to the original dataframe
# cleaned_data['cluster'] = clusters
#
# # Plot the clusters
# plt.figure(figsize=(10, 6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x')
# plt.title('KMeans Clustering (PCA-reduced Data)')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()
#
# # Display cluster centers in the original feature space
# cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
# centers_df = pd.DataFrame(cluster_centers, columns=X.columns)
# print("Cluster centers in original feature space:")
# print(centers_df)
