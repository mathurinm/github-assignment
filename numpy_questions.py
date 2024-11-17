"""
This module contains functions related to numpy operations and examples.

Functions:
- max_index: Returns the index of the maximum value in a 2D numpy array.
- wallis_product: Approximates the value of pi.
"""

import numpy as np


def max_index(X):
    """Return the index of the maximum in a numpy array.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input array.

    Returns
    -------
    (i, j) : tuple(int)
        The row and column index of the maximum.

    Raises
    ------
    ValueError
        If the input is not a numpy array or
        if the shape is not 2D.
    """
    if len(np.shape(X)) != 2:
        raise ValueError('Shape of X is not 2D')
    if not isinstance(X, np.ndarray):
        raise ValueError('X is not a numpy array')

    # Use np.argmax to find the index of the maximum value
    max_indices = np.argmax(X)

    # Convert the flattened index to 2D indices (row, column)
    i, j = np.unravel_index(max_indices, X.shape)

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    Parameters
    ----------
    n_terms : int
        Number of steps in the Wallis product. Note that `n_terms=0` will
        consider the product to be `1`.

    Returns
    -------
    pi : float
        The approximation of order "n_terms" of pi using the Wallis product.
    """
    if n_terms == 0:
        return 1

    prod = [(4 * i ** 2) / (4 * i ** 2 - 1) for i in range(1, n_terms + 1)]
    prod = np.array(prod)

    result = np.prod(prod)

    return 2 * result
