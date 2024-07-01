import pandas as pd
from math import radians, sin, cos, sqrt, atan2

university_gates_locations = [[31.263289, 34.806142],[31.261085, 34.800446],[31.263797, 34.798292],[31.265186, 34.802677]]


def haversine(loc1, loc2):
    R = 6371  # Radius of the Earth in kilometers
    lat1, lon1 = radians(loc1[0]), radians(loc1[1])
    lat2, lon2 = radians(loc2[0]), radians(loc2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


def get_df_with_university_distance(df):
    df['university_distance'] = 0.0
    for index, row in df.iterrows():
        loc = [row['latitude'], row['longitude']]
        university_distance = float('inf')
        for gates in university_gates_locations:
            d = haversine(loc, gates)
            if d < university_distance:
                university_distance = d
        df.at[index, 'university_distance'] = max(0, round(2 - university_distance,2)) if university_distance < 2 else 0
    return df


def get_df_with_distance(df, column_string, coor):
    df[column_string] = 0.0
    for index, row in df.iterrows():
        loc = [row['latitude'], row['longitude']]
        d = haversine(loc, coor)
        df.at[index, column_string] = max(0, round(2 - d,2)) if d < 2 else 0
    return df


file_path = 'outliers_cleaned_data.csv'
df = pd.read_csv(file_path)
df = get_df_with_university_distance(df)

# Define the locations with their respective names and coordinates
locations = {
    "central_station_distance": [31.243426, 34.796892],
    "sami_shamoon_distance": [31.250000, 34.789415],
    "soroka_distance": [31.257832, 34.800936],
    "north_train_distance": [31.262032, 34.808998],
    "center_train_distance": [31.243528, 34.798789],
    "grand_mall_distance": [31.250714, 34.772081],
    "old_city_distance": [31.240988, 34.787158]
}

# Update the DataFrame with distances for each location
for name, coordinates in locations.items():
    df = get_df_with_distance(df, name, coordinates)

df.to_csv("outliers_cleaned_data.csv", index=False)

'''

##Columns##
price
square_meters
rooms
floor
images_count
is_promoted
condition
latitude
longitude
price_per_sqm
Id

important_places = ["Ben-Gurion University", "Central_station","Sami-Shamoon"
                        ,"Soroka","North train","Center train"
                        "Grand mall","Old city" ]


'''