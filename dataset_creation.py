import pandas as pd
from utils import *

['feed_source', 'HomeTypeID_text', 'date_added' , 'search_text']

try:
    raw_df = pd.read_csv('Yad2Dataset.csv', encoding='utf-8')
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    raise

clean_df = pd.DataFrame()

# Continuous: Price, Square Meters
clean_df['price'] = raw_df['price'].str.replace('₪', '').str.replace(',', '')
clean_df['price'] = clean_df['price'].apply(lambda x: extract_numbers(x) if pd.notnull(x) else 0)
clean_df['square_meters'] = raw_df['square_meters'].apply(lambda x: extract_numbers(x) if pd.notnull(x) else 0)

# Discrete: Rooms, Floor, Images Count
clean_df['rooms'] = raw_df['line_1'].apply(lambda x: extract_numbers(x) if pd.notnull(x) else 0)
clean_df['floor'] = raw_df['line_2'].apply(lambda x: extract_numbers(x) if pd.notnull(x) else 0)
clean_df['images_count'] = raw_df['images_count'].apply(lambda x: extract_numbers(x) if pd.notnull(x) else 0)

# Ordinary: is_promoted, condition
clean_df['is_promoted'] = raw_df['ad_highlight_type'].apply(lambda x: ad_highlight[x] if pd.notnull(x) else 0)
clean_df['condition'] = raw_df['AssetClassificationID_text'].apply(lambda x: condition_dict[x] if pd.notnull(x) else 0)

# Coordinates
clean_df['latitude'], clean_df['longitude'] = zip(*raw_df['coordinates'].apply(lambda x: extract_coordinates(x) if pd.notnull(x) else (0, 0)))


# #############################################
# Feature Engineering
clean_df['price_per_sqm'] = clean_df['price']/clean_df['square_meters']



# #############################################
# Feature Engineering


clean_df.to_csv('cleaned_data.csv', index=False)
print(clean_df)

