import numpy as np

def plott(ax, X, Y, model, title, rmse):
    # print(model[1])
    ax.scatter(X, Y)
    ax.set_title('{} \nFinale RMSE: {:.2f}'.format(title, rmse))
    ax.plot(X, np.dot(X, model[0]) + model[1], c='g')
