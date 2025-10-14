import numpy as np
import pandas as pd

from sklearn import linear_model
from matplotlib import pyplot as plt

from service.preprocessing import preprocessing
from models.SGD import stochastic_gradient_descent
from models.OLS_lr import ols_custom
from models.metrics import mse, r2, rmse

# Y = w1 * accommodates + w2 * bedrooms + w3 * neighbourhood_group_cleansed + b

if __name__ == '__main__':

    PATH = 'listings.csv'
    data = preprocessing(PATH)

    train = data.train
    features = train[['accommodates',]]
    # features = pd.get_dummies(features, columns=['neighbourhood_group_cleansed'], dtype=int)
    X = features.values
    Y = train['price'].values
    sgd_custom_params = stochastic_gradient_descent(X, Y)
    ols_custom_params = ols_custom(X, Y)

    sgd_sk_model = linear_model.LinearRegression()
    sgd_sk_model.fit(X, Y)

    print(sgd_custom_params)
    print(ols_custom_params)
    print([sgd_sk_model.coef_, sgd_sk_model.intercept_])
    rmse_sgd_custom = rmse(X, Y, sgd_custom_params)
    rmse_ols_custom = rmse(X, Y, ols_custom_params)
    rmse_scikit = rmse(X, Y, [sgd_sk_model.coef_, sgd_sk_model.intercept_])
    print('SGD custom RMSE: {:.2f}'.format(rmse_sgd_custom))
    print('OLS custom RMSE: {:.2f}'.format(rmse_ols_custom))

    print('SK RMSE: {:.2f}'.format(rmse_scikit))

    if len(X[0]) == 1:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].scatter(X, Y)
        axes[0].set_title('Self made SGD model.\nFinale RMSE: {:.2f}'.format(rmse_sgd_custom))
        axes[1].scatter(X, Y)
        axes[1].set_title('Scikit OLS model.\nFinale RMSE: {:.2f}'.format(rmse_scikit))

        axes[0].plot(X, np.dot(X, sgd_custom_params[0]) + sgd_custom_params[1], c='g')

        axes[1].plot(X, np.dot(X, sgd_sk_model.coef_) + sgd_sk_model.intercept_, c='r')

        plt.show()
