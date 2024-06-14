import pandas as pd
import re

raw_df = pd.read_csv('Yad2Dataset.csv', encoding='utf-8')
['feed_source', 'HomeTypeID_text', 'images', 'date_added', 'coordinates', 'ad_highlight_type'
                                                                          'square_meters',
 'AssetClassificationID_text', 'images_count', 'search_text']
clean_df = pd.DataFrame()


def row_to_int(row):
    return pd.to_numeric(row, errors='coerce').fillna(0).astype(int)


def extract_numbers(s):
    if not s:
        return 0
    numeric_string = re.sub(r'\D', '', str(s))
    if numeric_string:
        return int(numeric_string)
    else:
        return 0


# Price - continuous variable
clean_df['price'] = raw_df['price'].str.replace('₪', '').str.replace(',', '')
clean_df['price'] = clean_df['price'].apply(lambda x: extract_numbers(x) if pd.notnull(x) else 0)

# Rooms
clean_df['rooms'] = raw_df['line_1'].apply(lambda x: extract_numbers(x) if pd.notnull(x) else 0)

# Floor
clean_df['floor'] = raw_df['line_2'].apply(lambda x: extract_numbers(x) if pd.notnull(x) else 0)

# Square Meters
clean_df['square_meters'] = raw_df['square_meters'].apply(lambda x: extract_numbers(x) if pd.notnull(x) else 0)

# Condition
condition_dict = {
    'חדש מקבלן (לא גרו בנכס)': 1,
    'חדש (גרו בנכס)': 2,
    'משופץ': 3,
    'במצב שמור': 4,
    'דרוש שיפוץ': 5
}
clean_df['condition'] = raw_df['AssetClassificationID_text'].apply(lambda x: condition_dict[x] if pd.notnull(x) else 0)


print(clean_df)
