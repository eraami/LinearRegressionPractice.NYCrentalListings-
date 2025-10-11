import numpy as np

from sklearn import linear_model


from service.preprocessing import preprocessing
from models.SGD import stochastic_gradient_descent
from models.metrics import mse

# Y = w1 * accommodates + w2 * bedrooms + w3 * neighbourhood_group_cleansed + b

if __name__ == '__main__':

    PATH = 'listings.csv'
    data = preprocessing(PATH)

    train = data.train
    X = train[['accommodates', 'bedrooms']].values
    Y = train['price'].values

    params = stochastic_gradient_descent(X, Y)

    sgd_sk_model = linear_model.LinearRegression()
    sgd_sk_model.fit(X, Y)

    print(sgd_sk_model.coef_, sgd_sk_model.intercept_)
    print(params)

    print('MSE custom: ', mse(X, Y, params))
    print('MSE sk: ', mse(X, Y, [sgd_sk_model.coef_, sgd_sk_model.intercept_]))
