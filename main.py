from common import read
from cost import squareLossCost, gradient
from grad_descent import gradient_descent

import numpy as np
from matplotlib import pyplot as plt

data = np.array(read('dataset/prices.csv'))

X, y = data[:, 0:-1], data[:, -1]

print(X, y)

res, history = gradient_descent(X, y, squareLossCost, gradient, 0.03, 0.0001, 100000)

print(res, len(history['cost']), history['cost'])
w, b = history['w'][-1], history['b'][-1]


x_min, x_max = np.min(X) - 1, np.max(X) + 1

plt.scatter(X, y, c='g', marker='x')
plt.plot([x_min, x_max], list(map(lambda x: x * w[0] + b, [x_min, x_max])), c='r')

plt.savefig('res.png')


