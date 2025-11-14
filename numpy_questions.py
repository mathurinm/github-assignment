"""Numpy utility functions."""

import numpy as np

def max_index(X):
    """Return the indices of the maximum value in a 2D array.

    Parameters
    ----------
    X : np.ndarray, shape (n_rows, n_cols)
        Input 2D array.

    Returns
    -------
    i, j : int, int
        Row and column indices of the maximum element.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    max_idx = np.argmax(X)
    i, j = np.unravel_index(max_idx, X.shape)
    return i, j

def wallis_product(n_terms):
    """Compute the Wallis product approximation of pi.

    Parameters
    ----------
    n_terms : int
        Number of terms in the product.

    Returns
    -------
    float
        Approximation of pi using the Wallis formula.
    """
    if n_terms == 0:
        return 1.0
    product = 1.0
    for k in range(1, n_terms + 1):
        product *= (4 * k**2) / (4 * k**2 - 1)
    return 2 * product
