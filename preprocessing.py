from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass
class PreprocessedData:
    train: pd.DataFrame
    test: pd.DataFrame
    predict: pd.DataFrame


def to_num(s: str):
    if ',' in s:
        s = s.replace(',', '')
    return float(s[1:]) if isinstance(s, str) else float(s)

def add_values_to_dataset(dataset, value_name, mean_value_set):
    dataset[value_name] = dataset[value_name].fillna(
        dataset['accommodates'].map(mean_value_set)
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

def preprocessing() -> PreprocessedData:

    df = pd.read_csv('listings.csv', usecols=use_cols + ['room_type'])

    data_mask = (df['room_type'] == 'Entire home/apt')
    df = df.loc[data_mask, use_cols].copy()

    na_price_data_mask = (df['price'].isna())

    labeled_dataset = df.loc[~na_price_data_mask, use_cols].copy()
    predict_dataset = df.loc[na_price_data_mask, use_cols].copy()

    labeled_dataset['price'] = labeled_dataset['price'].apply(to_num)

    train_dataset, test_dataset = split_data_set(labeled_dataset, 0.8)

    learning_mean_value_set = train_dataset.groupby('accommodates')['bedrooms'].mean()
    add_values_to_dataset(train_dataset, 'bedrooms', learning_mean_value_set)
    add_values_to_dataset(test_dataset, 'bedrooms', learning_mean_value_set)
    add_values_to_dataset(predict_dataset, 'bedrooms', learning_mean_value_set)

    return PreprocessedData(train_dataset, test_dataset, predict_dataset)


