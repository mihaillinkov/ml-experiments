import numpy as np


def gradient_descent(cost_function, grad_function, X, y, theta, alpha=0.3, p=0.0001, max_iterations=100):
    costs = [cost_function(X, y, theta)]
    history = [theta]

    converged = 1
    for i in range(max_iterations):
        theta = theta - alpha * grad_function(X, y, theta)
        history.append(np.copy(theta))
        costs.append(cost_function(X, y, theta))
        if abs(costs[-1] - costs[-2]) < p:
            break
    else:
        converged = 0

    return converged, theta, history, costs


def norm(M, rng):
    mu = np.mean(M, 0)
    std = np.std(M, 0)

    normM = M.copy()

    for c in rng:
        normM[:, c] = (normM[:, c] - mu[c]) / std[c]

    return normM
