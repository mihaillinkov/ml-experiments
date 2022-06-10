from matplotlib import pyplot as plt
from csv import reader
import numpy as np
from common import gradient_descent, norm


def plot_points(x, y):
    plt.scatter(x, y, c='g', marker='x', linewidths=1)
    plt.margins(0.2)


def plot_h(theta, minx, maxx):
    diff = (maxx - minx) * 0.1
    x = np.arange(minx - diff, maxx + diff, 0.1)
    y = [theta[0] + theta[1] * v for v in x]
    plt.plot(x, y, c='r', linewidth=0.5)


def cost(X, y, theta):
    m = theta.shape[0]
    return np.sum((X @ theta - y) ** 2) / (2 * m)


def grad(X, y, theta):
    m = theta.shape[0]
    return (X.T @ (X @ theta - y)) / m


def plot(x, y, theta, name):
    plt.clf()
    minx, maxx = min(x), max(x)
    plot_points(x, y)
    plot_h(theta, minx, maxx)
    plt.savefig(f'lin-reg/{name}.png')


def plotcountrour(X, y, history):
    theta1, theta2 = np.meshgrid(np.arange(20, 50, 1), np.arange(10, 40, 1))
    theta1 = theta1.flatten()
    theta2 = theta2.flatten()
    cst = [[cost(X, y, np.array([[t1], [t2]])) for t1 in theta1] for t2 in theta2]
    plt.clf()
    plt.contour(theta1, theta2, cst, levels=np.arange(0, 240, 20))
    plt.scatter([h[0][0] for h in history], [h[1][0] for h in history], c='r', marker='x')

    plt.savefig("lin-reg/cont.png")


with open('prices.csv', 'r') as file:
    prices = [list(map(float, row)) for row in reader(file)]
    X = np.array([[1, *row[0: -1]] for row in prices])
    y = np.array([[row[-1]] for row in prices])

    n = X.shape[1]
    theta = np.array([[0] for _ in range(n)])
    plot(X[:, 1], y[:, 0], theta[:, 0], 'origin')
    normX = norm(X, range(1, n))


    converged, theta, history, costs = gradient_descent(cost, grad, normX, y, theta)

    print(theta)
    plotcountrour(normX, y, history)
    plot(normX[:, 1], y[:, 0], theta[:, 0], 'final')