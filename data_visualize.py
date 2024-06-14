import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import image as mpimg

matplotlib.use('tkagg')

try:
    cleaned_data = pd.read_csv('cleaned_data.csv', encoding='utf-8')
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    raise

"""
# Distribution of Prices (Filtered)
filtered_data = cleaned_data[(cleaned_data['price'] < 20000000) & (cleaned_data['price'] > 1)]
sns.displot(filtered_data['price'], kde=True, bins=150)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Prices (Filtered)')
plt.show()
"""

# Apartments on the maps
img = mpimg.imread('BeershevaMap.png')
filtered_data = cleaned_data[(cleaned_data['latitude'] > 1) & (cleaned_data['longitude'] > 1)]
lat_min, lat_max = 34.7, 34.8
lon_min, lon_max = 31.2, 31.3
plt.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max])

plt.scatter(filtered_data['latitude'], filtered_data['longitude'], c=filtered_data['price_per_sqm'], cmap='viridis', alpha=0.5)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Geographical Distribution of Prices')
plt.show()
