"""
This module contains the solutions for the exercises.

It has functions to find the index of the max in an array and find pi.
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
    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a numpy array")

    if X.ndim != 2:
        raise ValueError("Input array must be 2D")

    # Trova l'indice piatto del massimo
    flat_index = np.argmax(X)

    # Usa divmod per ottenere riga e colonna
    i, j = divmod(flat_index, X.shape[1])

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : int
        Number of steps in the Wallis product. Note that `n_terms=0` will
        consider the product to be `1`.

    Returns
    -------
    pi : float
        The approximation of order `n_terms` of pi using the Wallis product.
    """
    if n_terms == 0:
        return 1.0

    product = 1.0
    for n in range(1, n_terms + 1):
        term = (4 * n ** 2) / ((4 * n ** 2) - 1)
        product *= term

    return 2 * product
