import numpy as np


def max_index(X):
    i = 0
    j = 0

    # Checking if the input is a 2D np array.
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise ValueError("Input must be a 2D numpy array")

    # Get the index of the maximum value
    max_index = np.argmax(X)  # By default, the index is in the flattened array
    num_rows, num_cols = X.shape

    i = max_index // num_cols
    j = max_index % num_cols

    return i, j


def wallis_product(n_terms):

    if n_terms == 0:
        return 2.0

    pi = 2.0  # We initialize pi.

    for i in range(1, n_terms + 1):
        pi *= 4.0 * i**2 / (4.0 * i**2 - 1)  # Wallis product.

    return pi
