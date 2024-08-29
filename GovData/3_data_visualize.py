import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import image as mpimg

matplotlib.use('tkagg')

try:
    cleaned_data = pd.read_csv('../Yad2/clean_gov_dataset.csv', encoding='utf-8')
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
img = mpimg.imread('../BeershevaMap.png')
lat_min, lat_max = 34.75, 34.82
lon_min, lon_max = 31.215, 31.3
plt.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max])

plt.scatter(cleaned_data['latitude'], cleaned_data['longitude'], c=cleaned_data['price_per_sqm'], cmap='viridis',
            alpha=0.5)
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
cleaned_data_corr = cleaned_data[['price_per_sqm', 'price', 'square_meters',
                                  'rooms', 'floor',
                                  'is_ground', 'new_building',
                                   'is_kottage', 'has_yard',
                                  'university_distance',
                                   'central_station_distance', 'sami_shamoon_distance', 'soroka_distance',
                                   'north_train_distance', 'center_train_distance', 'grand_mall_distance',
                                   'old_city_distance']].corr()

dataplot = sns.heatmap(cleaned_data_corr, cmap="YlGnBu", annot=True)
plt.xticks(rotation=40, horizontalalignment='right')
plt.show()

#  Binary features distribution
columns_names = ["safe_room", "elevator", "parking", "bars",
                 "air_conditioner", "accessible", "furniture"]
features_to_distribute = cleaned_data[columns_names]
fig, axs = plt.subplots(2, 4, figsize=(15, 10))
colors_dict = {0: 'red', 1: 'green'}

for i, col_name in enumerate(columns_names):
    row = i // 4  # Determine row index
    col = i % 4  # Determine column index

    counts = features_to_distribute[col_name].value_counts()
    labels = counts.index
    sizes = counts.values
    colors = [colors_dict[idx] for idx in labels]

    axs[row, col].pie(sizes, labels=labels, colors=colors, autopct='%.0f%%')
    axs[row, col].set_title(col_name)

plt.tight_layout()
plt.show()

#  Ordinary features distribution
columns_names = ['is_promoted', 'floor', 'rooms', 'condition']
features_to_distribute = cleaned_data[columns_names]
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

for i, col_name in enumerate(columns_names):
    row = i // 2  # Determine row index
    col = i % 2   # Determine column index
    value_counts = features_to_distribute[col_name].value_counts()

    axs[row, col].bar(value_counts.index, value_counts.values)
    axs[row, col].set_title(col_name)

plt.tight_layout()
plt.show()
