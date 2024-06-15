import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import image as mpimg

matplotlib.use('tkagg')

try:
    cleaned_data = pd.read_csv('outliers_cleaned_data.csv', encoding='utf-8')
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    raise


# Distribution of Prices
sns.displot(cleaned_data['price'], kde=True, bins=150)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Prices')
plt.show()


# Apartments on the maps
img = mpimg.imread('BeershevaMap.png')
lat_min, lat_max = 34.7, 34.8
lon_min, lon_max = 31.2, 31.3
plt.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max])

plt.scatter(cleaned_data['latitude'], cleaned_data['longitude'], c=cleaned_data['price_per_sqm'], cmap='viridis', alpha=0.5)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Geographical Distribution of Prices')
plt.show()


# Distribution of price_per_sqm (Filtered)
sns.displot(cleaned_data['price_per_sqm'], kde=True, bins=150)
plt.xlabel('price_per_sqm')
plt.ylabel('Frequency')
plt.title('Distribution of price_per_sqm (Filtered)')
plt.show()


# Correlation Bitmap
cleaned_data_corr = cleaned_data.drop(columns=['latitude', 'longitude']).corr()
dataplot = sns.heatmap(cleaned_data_corr, cmap="YlGnBu", annot=True)
plt.show()
