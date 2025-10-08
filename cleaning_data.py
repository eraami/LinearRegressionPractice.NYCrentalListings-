import pandas as pd

"""
neighbourhood_group_cleansed
latitude *
longitude *import pandas as pd


neighbourhood_group_cleansed
latitude *
longitude *
room_type - 
accommodates
bathrooms
bedrooms
amenities ?
price
review_scores_rating

-> neighbourhood_group_cleansed + accommodates + bathrooms + bedrooms + review_scores_rating = price
"""

def to_num(s: str):
    if ',' in s:
        s = s.replace(',', '')
    return float(s[1:]) if isinstance(s, str) else float(s)

use_cols = [
    'neighbourhood_group_cleansed',
    # 'latitude',
    # 'longitude',
    'accommodates',
    'bathrooms',
    'bedrooms',
    # 'amenities',
    'price',
    'review_scores_rating'
]

df = pd.read_csv('listings.csv', usecols=use_cols + ['room_type'])

filter_mask_train_data = (df['room_type'] == 'Entire home/apt') & (df['price'])

train_data_filtered = df.loc[filter_mask_train_data, use_cols]
train_data_filtered['price'] = train_data_filtered['price'].apply(to_num)

print(train_data_filtered)

# print(X, Y)
