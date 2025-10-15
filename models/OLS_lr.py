import numpy as np


def ols_custom(X, Y):
    """Calculating Linear Regression model by ordinary least squares"""

    # Add ones-column to calc intercept
    X = np.column_stack((np.ones(len(X)), X))
    w = np.linalg.inv(X.T @ X) @ X.T @ Y

    return [w[1:], w[0]] # Project style model representation
