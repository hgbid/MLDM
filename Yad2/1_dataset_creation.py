import pandas as pd
from utils import *


try:
    raw_df = pd.read_csv('../Yad2Dataset.csv', encoding='utf-8')
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
clean_df['is_ground'] = (clean_df['floor'] == 0).astype(int)

# Ordinary: is_promoted, condition
clean_df['is_promoted'] = raw_df['ad_highlight_type'].apply(lambda x: ad_highlight[x] if pd.notnull(x) else 0)
clean_df['condition'] = raw_df['AssetClassificationID_text'].apply(lambda x: condition_dict[x] if pd.notnull(x) else 0)

# Coordinates
clean_df['latitude'], clean_df['longitude'] = zip(*raw_df['coordinates'].apply(lambda x: extract_coordinates(x) if pd.notnull(x) else (0, 0)))

clean_df['date'] = 0

# print(raw_df['feed_source'])
# print(raw_df['date_added'])

# #############################################

# Feature Engineering
clean_df['price_per_sqm'] = (clean_df['price']/clean_df['square_meters'])

clean_df['HomeTypeID_text'] = raw_df['HomeTypeID_text'].fillna('')
clean_df['is_kottage'] = clean_df['HomeTypeID_text'].str.contains('קוטג').astype(int)
clean_df['is_penthouse'] = clean_df['HomeTypeID_text'].str.contains('גג').astype(int)
clean_df['has_yard'] = clean_df['HomeTypeID_text'].str.contains('גן').astype(int)
values_to_remove = ['חניה', 'מגרשים', 'כללי']
clean_df = clean_df[~clean_df['HomeTypeID_text'].isin(values_to_remove)]
clean_df = clean_df.drop(columns=["HomeTypeID_text"])
clean_df['HomeTypeID_text'] = raw_df['HomeTypeID_text']


# DM from text: safe_room, elevator, parking, bars, air_conditioning, air_conditioner, accessible, furniture
features_df = raw_df['search_text'].apply(lambda x: pd.Series(get_data_from_search_text(x)))
clean_df = pd.concat([clean_df, features_df], axis=1)

clean_df["air_conditioner"] = clean_df["air_conditioner"] | clean_df["air_conditioning"]
clean_df = clean_df.drop(columns=["air_conditioning"])

clean_df['new_building'] = clean_df["safe_room"] | clean_df["elevator"]


clean_df.to_csv('./yad2_dataset.csv', index=False)

