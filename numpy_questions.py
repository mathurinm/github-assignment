import numpy as np


def max_index(X):
    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")

    max_idx = np.argmax(X)
    i, j = np.unravel_index(max_idx, X.shape)
    return i, j


def wallis_product(n_terms):
    if n_terms == 0:
        return 1.0

    product = 1.0
    for k in range(1, n_terms + 1):
        product *= (4 * k ** 2) / (4 * k ** 2 - 1)

    return 2 * product
