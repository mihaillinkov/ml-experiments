import numpy as np


def gradient_descent(X, y, cost, grad, alpha, precision, max_iter):
    w = np.zeros(X.shape[1])
    b = 0
    history = {
        'cost': [cost(w, b, X, y)],
        'w': [w],
        'b': [b]
    }

    for i in range(max_iter):
        dw, db = grad(w, b, X, y, alpha)
        w, b = w - dw, b - db

        history['cost'].append(cost(w, b, X, y))
        history['w'].append(w)
        history['b'].append(b)

        if abs(history['cost'][-1] - history['cost'][-2]) < precision:
            break
    else:
        return 0, history

    return 1, history
