import pandas as pd

from sklearn import linear_model
from matplotlib import pyplot as plt

from service.preprocessing import preprocessing
from models.SGD import stochastic_gradient_descent
from models.OLS_lr import ols_custom
from models.metrics import rmse
from plotting.plott import plott


# Y = w1 * accommodates + w2 * bedrooms + w3 * neighbourhood_group_cleansed + b
# model = [[slopes,], intercept]


def predict(X, model):
    predictions = X * model[0] + model[1]
    return predictions


if __name__ == '__main__':

    PATH = 'listings.csv'
    X_LABELS = ['accommodates',]
    PRINT = False

    data = preprocessing(PATH)

    # Data selection
    train_data = data.train
    predict_features = data.predict[X_LABELS].values
    train_features = train_data[X_LABELS]

    # Refactor STRING feature to multi columns INT
    if len(X_LABELS) > 1:

        train_features = pd.get_dummies(train_features, columns=['neighbourhood_group_cleansed'], dtype=int)
        predict_features = pd.get_dummies(predict_features, columns=['neighbourhood_group_cleansed'], dtype=int)

    # Train data
    X = train_features.values # features
    Y = train_data['price'].values  # labels

    # Custom (self-made) Linreg models -> [[slopes,], intercept]
    sgd_custom_params = stochastic_gradient_descent(X, Y)
    ols_custom_params = ols_custom(X, Y)

    # Scikit-learn models
    sgd_sk_model = linear_model.LinearRegression()
    ols_sk_model = linear_model.SGDRegressor()
    ols_sk_model.fit(X, Y)
    sgd_sk_model.fit(X, Y)

    if PRINT:
        print(sgd_custom_params)
        print(ols_custom_params)
        print([sgd_sk_model.coef_, sgd_sk_model.intercept_])
        print([ols_sk_model.coef_, ols_sk_model.intercept_])

    rmse_sgd_custom = rmse(X, Y, sgd_custom_params)
    rmse_ols_custom = rmse(X, Y, ols_custom_params)
    rmse_scikit = rmse(X, Y, [sgd_sk_model.coef_, sgd_sk_model.intercept_])
    rmse_scikit_ols = rmse(X, Y, [ols_sk_model.coef_, ols_sk_model.intercept_])

    if PRINT:
        print('SGD custom RMSE: {:.2f}'.format(rmse_sgd_custom))
        print('OLS custom RMSE: {:.2f}'.format(rmse_ols_custom))
        print('OLS sk RMSE: {:.2f}'.format(rmse_scikit_ols))
        print('SK RMSE: {:.2f}'.format(rmse_scikit))

    # 2d graphs
    if len(X[0]) == 1:
        fig, axes = plt.subplots(2, 2, figsize=(25, 10))

        plott(axes[0][0], X, Y, sgd_custom_params, 'Self made SGD model', rmse_sgd_custom)
        plott(axes[0][1], X, Y, ols_custom_params, 'Self made OLS model', rmse_ols_custom)

        plott(axes[1][0], X, Y, [sgd_sk_model.coef_, sgd_sk_model.intercept_], 'Scikit SGD model', rmse_scikit)
        plott(axes[1][1], X, Y, [ols_sk_model.coef_, ols_sk_model.intercept_], 'Scikit OLS model', rmse_scikit_ols)

        plt.show()

    # Predictions
    sgd_custom_predictions = predict(predict_features, sgd_custom_params)
    sk_predictions = predict(predict_features, [sgd_sk_model.coef_, sgd_sk_model.intercept_])
    plt.scatter(predict_features, sgd_custom_predictions, marker='^', c='g', s=10, label='Custom predictions')
    plt.scatter(predict_features, sk_predictions, marker='s', c='r', s=10, label='Scikit predictions')
    plt.legend()
    plt.show()
