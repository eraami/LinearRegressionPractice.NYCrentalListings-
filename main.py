import numpy as np
import pandas as pd
from pprint import pprint

from sklearn import linear_model


from service.preprocessing import preprocessing
from models.SGD import stochastic_gradient_descent
from models.OLS_lr import ols_custom
from models.metrics import mse, r2

# Y = w1 * accommodates + w2 * bedrooms + w3 * neighbourhood_group_cleansed + b

if __name__ == '__main__':

    PATH = 'listings.csv'
    data = preprocessing(PATH)

    train = data.train
    features = train[['accommodates', 'neighbourhood_group_cleansed']]
    features = pd.get_dummies(features, columns=['neighbourhood_group_cleansed'], dtype=int)
    print(features.head())
    X = features.values
    Y = train['price'].values

    sgd_custom_params = stochastic_gradient_descent(X, Y)
    # ols_custom_params = ols_custom(X, Y)

    sgd_sk_model = linear_model.LinearRegression()
    sgd_sk_model.fit(X, Y)

    pprint(sgd_custom_params)
    # print(ols_custom_params)
    pprint([sgd_sk_model.coef_, sgd_sk_model.intercept_])

    print('SGD custom MSE: {:.2f}'.format(r2(X, Y, sgd_custom_params)))
#     print('OLS custom MSE: ', mse(X, Y, ols_custom_params))
    print('SK MSE: {:.2f}'.format(r2(X, Y, [sgd_sk_model.coef_, sgd_sk_model.intercept_])))
