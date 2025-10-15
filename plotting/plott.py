import numpy as np


def plott(ax, X, Y, model, title, rmse):
    ax.scatter(X, Y)
    title = '{}'.format(title)
    if model:
        title += '\nFinale RMSE: {:.2f}'.format(rmse)
        ax.plot(X, np.dot(X, model[0]) + model[1], c='g')
    ax.set_title(title)
