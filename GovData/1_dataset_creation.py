import pandas as pd
from gov_utils import *
import chardet

# Detect the encoding of the CSV file
with open('GovDataset.csv', 'rb') as file:
    result = chardet.detect(file.read())
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")

try:
    raw_df = pd.read_csv('GovDataset.csv', encoding=encoding)
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
clean_df['is_ground'] = (clean_df['floor'] ==0).astype(int)

raw_df['DEALDATE'] = pd.to_datetime(raw_df['DEALDATE'])
specific_date = pd.to_datetime('2024-06-03')
clean_df['date'] = ((specific_date - raw_df['DEALDATE']).dt.days)//30

# Coordinates
clean_df['latitude'], clean_df['longitude'] = raw_df['lat'], raw_df['long']
clean_df['FULLADRESS'] = raw_df['FULLADRESS'].dropna()
clean_df = clean_df.drop(columns=["FULLADRESS"])


# Feature Engineering
clean_df['price_per_sqm'] = clean_df['price'] / clean_df['square_meters']

clean_df['new_building'] = raw_df['BUILDINGYEAR'].apply(lambda x: 1 if (pd.notnull(x) and x > 1990) else 0)


clean_df['DEALNATUREDESCRIPTION'] =raw_df['DEALNATUREDESCRIPTION'].fillna('')
values_to_remove = ['חנות','מחסנים','ללא תיכנון','משרד','קרקע למגורים','מסחרי + מגורים','תעשיה']
clean_df['is_kottage'] = clean_df['DEALNATUREDESCRIPTION'].str.contains('קוטג').astype(int)
# clean_df['is_penthouse'] = clean_df['DEALNATUREDESCRIPTION'].str.contains('גג').astype(int)
clean_df['has_yard'] = clean_df['DEALNATUREDESCRIPTION'].str.contains('גן').astype(int)
clean_df = clean_df[~clean_df['DEALNATUREDESCRIPTION'].isin(values_to_remove)]
clean_df = clean_df.drop(columns=["DEALNATUREDESCRIPTION"])

clean_df.to_csv('gov_dataset.csv', index=False, encoding='utf-8-sig')
print(clean_df)


