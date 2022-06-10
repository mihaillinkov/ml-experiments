from csv import reader
from matplotlib import pyplot as plt
import numpy as np
from common import gradient_descent, norm


def extract_y(y, target):
    return np.array([[1 if (v == target) else 0 for v in y]]).T


def cost(X, y, theta):
    m = y.shape[0]

    sig = sigmoid(X, theta)
    return (-1 / m * (y.T @ np.log(sig) + (1 - y.T) @ np.log(1 - sig)))[0][0]


def grad(X, y, theta):
    m = y.shape[0]
    return 1 / m * (X.T @ (sigmoid(X, theta) - y))


def sigmoid(X, theta):
    m = X.shape[0]
    h = np.power(np.array([[np.e] * m]).T, -(X @ theta))
    return 1 / (1 + h)


def plot(x1, x2, y, mapping):
    subplot = plt.subplot()
    c = list(zip(x1, x2, y))

    for i in mapping.keys():
        (name, m, col) = mapping[i]
        t = list(filter(lambda it: it[2] == i, c))
        subplot.scatter(
            list(map(lambda it: it[0], t)),
            list(map(lambda it: it[1], t)),
            c=col,
            marker=m,
            label=name
        )

    subplot.legend()
    return subplot


def plot_line(X, theta):
    minx, maxx = min(X[:, 1].flatten()), max(X[:, 1].flatten())
    miny, maxy = -2, 2

    x, y = np.meshgrid(np.linspace(minx, maxx, 100), np.linspace(miny, maxy, 100))
    z = calc(theta, x, y)

    plt.contour(x, y, z, [0])


def calc(theta, x1, x2):
    x1 = x1[0, :]
    x2 = x2[:, 0]

    z = np.zeros((x1.size, x2.size))
    for (r, x1v) in enumerate(x1):
        for (c, x2v) in enumerate(x2):
            z[r, c] = func(x1v, x2v, theta)

    return z.T


def func(x1, x2, theta):
    return theta[0, 0] + theta[1, 0] * x1 + theta[2, 0] * x2 + theta[3, 0] * x1 * x1 + theta[4, 0] * x2 * x2


mapping = {
    1: ['Yes', 'x', 'r'],
    0: ['No', 'o', 'y']}

with open('dataset/illness.csv', 'r') as illness:

    data = [list(map(float, row)) for row in reader(illness)]

    X = np.array([[1, *row[:-1]] for row in data])
    y = np.array([[int(row[-1])] for row in data])

    normX = norm(X, range(1, X.shape[1]))
    normX = np.append(normX, normX[:, [1]] ** 2, 1)
    normX = np.append(normX, normX[:, [2]] ** 2, 1)
    theta = np.array([[-1] * normX.shape[1]]).T

    converged, theta, history, costs = gradient_descent(cost, grad, normX, y, theta)

    print(costs)
    plt.clf()
    plot_line(normX, theta)
    plot(normX[:, 1].flatten(), normX[:, 2].flatten(), y.flatten(), {1: ['Yes', 'x', 'r'], 0: ['No', 'o', 'y']})
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig(f'log-reg/plot.png')


