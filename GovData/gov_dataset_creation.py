import pandas as pd
from gov_utils import *
import chardet

# Detect the encoding of the CSV file
with open('./GovDataset_updated.csv', 'rb') as file:
    result = chardet.detect(file.read())
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")

try:
    raw_df = pd.read_csv('./GovDataset_updated.csv', encoding=encoding)
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    raise

clean_df = pd.DataFrame()

# Continuous: Price, Square Meters
clean_df['price'] = raw_df['DEALAMOUNT']
clean_df['square_meters'] = raw_df['DEALNATURE']

# Discrete: Rooms, Floor, Images Count
clean_df['rooms'] = raw_df['ASSETROOMNUM']
clean_df['floor'] = raw_df['FLOORNO'].apply(lambda x: get_floor_number(x) if pd.notnull(x) else 0)

# Coordinates
clean_df['latitude'], clean_df['longitude'] = raw_df['lat'], raw_df['long']

# Feature Engineering
clean_df['price_per_sqm'] = clean_df['price'] / clean_df['square_meters']

clean_df.to_csv('cleaned_data_gov.csv', index=False, encoding='utf-8-sig')
print(clean_df)
