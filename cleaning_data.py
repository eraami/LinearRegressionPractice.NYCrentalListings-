import random

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
        dataset[group_to_mean].map(means)
    )

def split_data_set(dataset, percentage):

    rng = np.random.default_rng()
    mask = rng.random(len(dataset)) < percentage
    bigger_df = dataset[mask].copy()
    less_df = dataset[~mask].copy()
    return bigger_df, less_df


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

learn_dataset, test_dataset = split_data_set(train_test_dataset, 0.8)

add_values_to_dataset(predict_dataset, 'bedrooms', 'accommodates')
add_values_to_dataset(learn_dataset, 'bedrooms', 'accommodates')
add_values_to_dataset(test_dataset, 'bedrooms', 'accommodates')

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(learn_dataset)

