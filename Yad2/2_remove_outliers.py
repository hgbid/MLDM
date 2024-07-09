import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils import *

matplotlib.use('tkagg')

try:
    cleaned_data = pd.read_csv('../cleaned_data.csv', encoding='utf-8')
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    raise


def show_outliers(df):
    columns_names = df.columns.values
    num_plots = min(len(columns_names), 10)  # Ensure we don't plot more than available columns
    for i in range(num_plots):
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
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df


def apply_filters_and_count_removed_rows(df, filters):
    filtered_data = df.copy()
    for column in filters.keys():
        original_size = filtered_data.shape[0]
        filtered_data = apply_iqr_filter(filtered_data, column)
        filtered_size = filtered_data.shape[0]
        print_data_loss(original_size, filtered_size, column)
    return filtered_data


filters = {
    'price': {},
    'square_meters': {},
    'price_per_sqm': {},
}

initial_size = cleaned_data.shape[0]
print(f"Initial size of cleaned_data: {initial_size} rows")

outliers_removed_data = apply_filters_and_count_removed_rows(cleaned_data, filters)
show_outliers(outliers_removed_data)

print(f"Final size of cleaned_data after all filters: {outliers_removed_data.shape[0]} rows")


# Handle latitude and longitude
def fill_random_lat_lon(df, column):
    df.loc[df[column].isin([0, 1]), column] = np.nan
    print(df[column].isnull().sum())
    min_val = df[column].min()
    max_val = df[column].max()
    nan_indices = df[df[column].isna()].index
    random_values = np.random.uniform(min_val, max_val, len(nan_indices))
    df.loc[nan_indices, column] = random_values
    return df


outliers_removed_data = fill_random_lat_lon(outliers_removed_data, 'latitude')
outliers_removed_data = fill_random_lat_lon(outliers_removed_data, 'longitude')

# #############################################
# Add distant
clean_df = get_df_with_university_distance(outliers_removed_data)
for name, coordinates in locations.items():
    clean_df = get_df_with_distance(clean_df, name, coordinates)

clean_df.to_csv('clean_yad2_dataset.csv', index=False)
print("Saved cleaned data with added distances to 'outliers_cleaned_data_with_distance.csv'")
