"""
numpy_questions.py

This module contains functions related
to numpy operations and mathematical computations.

Functions
---------
max_index(X)
    Return the index of the maximum in a numpy array.

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
        If the input is not a numpy array or if the shape is not 2D.

wallis_product(n_terms)
    Implement the Wallis product to compute an approximation of pi.

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
        If the input is not a numpy array or if the shape is not 2D.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("Input is not a numpy array")

    if X.ndim != 2:
        raise ValueError("Input array must be 2D")

    # Find the indices of the maximum value
    max_index = np.unravel_index(np.argmax(X), X.shape)

    return max_index


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
    # Wallis product
    pi_approximation = 2.0
    for i in range(1, n_terms + 1):
        numerator = 4 * i**2
        denominator = 4 * i**2 - 1
        pi_approximation *= numerator / denominator

    if n_terms == 0:
        return 2.0

    return pi_approximation
