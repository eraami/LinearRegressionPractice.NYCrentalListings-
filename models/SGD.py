import random
import numpy as np

from matplotlib import pyplot as plt

from .metrics import mse


def stochastic_gradient_descent(X, Y, learning_rate=0.0001, epoches=10000, logging_epoches=None):

    # Setting initial params randomly
    slopes = np.array([random.random() for _ in range(len(X[0]))])
    intercept = random.random()

    # Save mse on every epoch
    mse_by_epoches = []

    for j in range(epoches):

        # Picking a random point
        i = random.randint(0, len(X) - 1)
        x = X[i]
        y = Y[i]

        # Calc prediction and error
        y_pred = np.dot(slopes, x) + intercept
        error = y - y_pred

        # Update params base on x and error between actual y and predicted
        slopes += x * error * learning_rate
        intercept += error * learning_rate

        mse_value = mse(X, Y, [slopes, intercept])
        mse_by_epoches.append(mse_value)

        if logging_epoches and j % logging_epoches == 0:
            print(mse_value)

    # MSE plot if needed
    if logging_epoches:
        plt.scatter(range(len(mse_by_epoches)), mse_by_epoches)
        plt.title('MSE to epoches at self made SGD')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.show()

    return slopes, intercept