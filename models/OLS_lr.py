import numpy as np


def ols_custom(X, Y):

    numerator = np.sum((X - X.mean()) * (Y - Y.mean()))
    denominator = np.sum((X - X.mean()) ** 2)

    slope = numerator / denominator
    intercept = Y.mean() - X.mean() * slope

    return [slope,], intercept