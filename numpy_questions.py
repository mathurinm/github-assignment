"""
Assignment - using numpy and making a PR.

This module implements two simple numerical functions: max_index and
wallis_product. These functions are tested using pytest.
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
    # Check type
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy ndarray.")

    # Check shape (must be 2D)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    # Find flat index of max and convert it to 2D indices
    flat_idx = np.argmax(X)
    i, j = np.unravel_index(flat_idx, X.shape)

    return int(i), int(j)


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
    if not isinstance(n_terms, (int, np.integer)) or n_terms < 0:
        raise ValueError("n_terms must be a non-negative integer.")

    # Empty product convention: product = 1 when n_terms = 0
    if n_terms == 0:
        return 1.0

    k = np.arange(1, n_terms + 1, dtype=float)
    terms = (2 * k) ** 2 / ((2 * k - 1) * (2 * k + 1))
    product = np.prod(terms)

    # Wallis product gives pi / 2, so multiply by 2
    return 2.0 * product
