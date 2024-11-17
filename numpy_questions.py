"""Module for numpy-based exercises.

This module contains functions to practice using numpy, such as:
- Finding the index of the maximum value in a 2D array.
- Approximating pi using the Wallis product.
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
    (i, j) : tuple of int
        The row and column index of the maximum.

    Raises
    ------
    ValueError
        If the input is not a numpy array or if the shape is not 2D.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    max_idx = np.unravel_index(np.argmax(X), X.shape)
    return max_idx


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
    if n_terms < 0:
        raise ValueError("n_terms must be a non-negative integer.")

    product = 1.0
    for n in range(1, n_terms + 1):
        term = (4 * n ** 2) / ((4 * n ** 2) - 1)
        product *= term

    return product * 2 if n_terms > 0 else 1.0
