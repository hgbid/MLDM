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


def show_outliers(df):
    columns_names = df.columns.values

    for i in range(len(columns_names)):
        plt.subplot(2, 5, i + 1)
        plt.boxplot(df[columns_names[i]])
        plt.title(columns_names[i])
    plt.show()


show_outliers(cleaned_data)

def print_data_loss(original_size, filtered_size, filter_name):
    loss = original_size - filtered_size
    print(f"Data loss after {filter_name}: {loss} rows")


def apply_filters_and_count_removed_rows(df, filters):
    filtered_data = df
    for filter_name, filter_params in filters.items():
        if 'min' in filter_params:
            min_val = filter_params['min']
            filtered_data = filtered_data[filtered_data[filter_name] > min_val]
        if 'max' in filter_params:
            max_val = filter_params['max']
            filtered_data = filtered_data[filtered_data[filter_name] < max_val]
        rows_removed = df.shape[0] - filtered_data.shape[0]
        print(f"Removed {rows_removed} rows with filter: {filter_name}")

    return filtered_data

filters = {
    'price': {'min': 1, 'max': 20000000},
    'square_meters': {'min': 1, 'max': 20000},
    'price_per_sqm': {'min': 1, 'max': 4000},
    'rooms': {'min': 0, 'max': 20},
    'latitude': {'min': 0},
    'longitude': {'min': 0}
}

initial_size = cleaned_data.shape[0]
print(f"Initial size of cleaned_data: {initial_size} rows")

outliers_removed_data = apply_filters_and_count_removed_rows(cleaned_data, filters)
show_outliers(outliers_removed_data)

print(f"Final size of cleaned_data after all filters: {outliers_removed_data.shape[0]} rows")

outliers_removed_data.to_csv('outliers_cleaned_data.csv', index=False)