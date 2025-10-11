import random

import numpy as np
from .metrics import mse


def stochastic_gradient_descent(X, Y, learning_rate=0.0001, epoches=10000, logging_epoches=None):
    slopes = np.array([random.random() for _ in range(len(X[0]))])
    intercept = random.random()
    for j in range(epoches):
        i = random.randint(0, len(X) - 1)
        x = X[i]
        y = Y[i]

        y_pred = np.dot(slopes, x) + intercept

        error = y - y_pred

        slopes += x * error * learning_rate
        intercept += error * learning_rate

        if logging_epoches and j % logging_epoches == 0:
            print(mse(X, Y, [slopes, intercept]))

    return slopes, intercept