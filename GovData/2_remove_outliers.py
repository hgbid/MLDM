import numpy as np
from scipy.stats import gaussian_kde

from utils import *
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
matplotlib.use('tkagg')

try:
    cleaned_data = pd.read_csv('../Yad2/gov_dataset.csv', encoding='utf-8')
    cleaned_data = cleaned_data.dropna()
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    raise

def show_outliers(df):
    columns_names = df.columns.values

    for i in range(7):
        plt.subplot(2, 5, i + 1)
        plt.boxplot(df[columns_names[i]])
        plt.title(columns_names[i])
    plt.show()

show_outliers(cleaned_data)

def print_data_loss(original_size, filtered_size, filter_name):
    loss = original_size - filtered_size
    print(f"Data loss after {filter_name}: {loss} rows")

def apply_iqr_filter(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.0 * IQR
    upper_bound = Q3 + 1.1 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df, lower_bound, upper_bound

def apply_filters_and_count_removed_rows(df, filters):
    filtered_data = df.copy()
    iqr_bounds = {}
    for column in filters.keys():
        original_size = filtered_data.shape[0]
        filtered_data, lower_bound, upper_bound = apply_iqr_filter(filtered_data, column)
        filtered_size = filtered_data.shape[0]
        print_data_loss(original_size, filtered_size, column)
        iqr_bounds[column] = (lower_bound, upper_bound)

    return filtered_data, iqr_bounds

filters = {
    'price': {},
    'square_meters': {},
    'price_per_sqm': {},
}

initial_size = cleaned_data.shape[0]
print(f"Initial size of cleaned_data: {initial_size} rows")

outliers_removed_data, iqr_bounds = apply_filters_and_count_removed_rows(cleaned_data, filters)


def fill_random_lat_lon_gaussian(df, column, min_val, max_val, mean, std_dev):
    # Set values outside the valid range to NaN
    df.loc[(df[column] < min_val) | (df[column] > max_val), column] = np.nan
    print(f"Number of NaN values before random filling in {column}: {df[column].isnull().sum()}")

    # Sample random values from a Gaussian distribution within the specified range
    nan_indices = df[df[column].isna()].index
    random_values = np.random.normal(mean, std_dev, len(nan_indices))

    # Ensure the sampled values are within the specified range
    random_values = np.clip(random_values, min_val, max_val)

    # Assign random values to NaN indices
    df.loc[nan_indices, column] = random_values

    return df


# Define the parameters for latitude and longitude
lat_min, lat_max = 31.23, 31.29
lon_min, lon_max = 34.75, 34.82

# Calculate mean and standard deviation for latitude and longitude
lat_mean = (lat_min + lat_max) / 2
lat_std = (lat_max - lat_min) / 4  # Arbitrary choice of std_dev; adjust as necessary

lon_mean = (lon_min + lon_max) / 2
lon_std = (lon_max - lon_min) / 4  # Arbitrary choice of std_dev; adjust as necessary

# Apply the function to latitude and longitude columns
outliers_removed_data = fill_random_lat_lon_gaussian(outliers_removed_data, 'latitude', lat_min, lat_max, lat_mean,
                                                     lat_std)
outliers_removed_data = fill_random_lat_lon_gaussian(outliers_removed_data, 'longitude', lon_min, lon_max, lon_mean,
                                                     lon_std)

show_outliers(outliers_removed_data)

print(f"Final size of cleaned_data after all filters: {outliers_removed_data.shape[0]} rows")



# #############################################
# Add distant
clean_df = get_df_with_university_distance(outliers_removed_data)
for name, coordinates in locations.items():
    clean_df = get_df_with_distance(clean_df, name, coordinates)

print("Saved cleaned data with added distances to 'clean_gov_dataset.csv'")
clean_df.to_csv('clean_gov_dataset.csv', index=False)



# Plotting the displot with IQR range
sns.displot(cleaned_data['price_per_sqm'], kde=True, bins=150)
plt.axvline(x=iqr_bounds['price_per_sqm'][0], color='red', linestyle='--', label='Lower Bound')
plt.axvline(x=iqr_bounds['price_per_sqm'][1], color='green', linestyle='--', label='Upper Bound')
plt.xlabel('price_per_sqm')
plt.ylabel('Frequency')
plt.title('Distribution of price_per_sqm (Filtered)')
plt.legend()
plt.show()
