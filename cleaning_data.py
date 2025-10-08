import pandas as pd
import numpy as np

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

def add_values_to_dataset(dataset, value, group_to_mean):
    means = dataset.groupby(group_to_mean)[value].mean()
    dataset[value] = dataset[value].fillna(
        predict_dataset[group_to_mean].map(means)
    )
    print(means)

use_cols = [
    'neighbourhood_group_cleansed',
    'latitude',
    'longitude',
    'accommodates',
    # 'bathrooms',
    'bedrooms',
    # 'amenities',
    'price',
    # 'review_scores_rating'
]

df = pd.read_csv('listings.csv', usecols=use_cols + ['room_type'])

data_mask = (df['room_type'] == 'Entire home/apt')
na_price_data_mask = (df['price'].isna())

df = df.loc[data_mask, use_cols]

train_test_dataset = df.loc[~na_price_data_mask, use_cols]
predict_dataset = df.loc[na_price_data_mask, use_cols]

train_test_dataset['price'] = train_test_dataset['price'].apply(to_num)

add_values_to_dataset(predict_dataset, 'bedrooms', 'accommodates')


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(predict_dataset[['accommodates', 'bedrooms']])

