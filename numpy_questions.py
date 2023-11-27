import numpy as np


def max_index(X):
    """
    Return the index of the maximum in a numpy array.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input array.

    Returns
    -------
    tuple of int
        The row and column index of the maximum.

    Raises
    ------
    ValueError
        If the input is not a numpy array or if the shape is not 2D.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    i, j = np.unravel_index(np.argmax(X), X.shape)
    return i, j


def wallis_product(n_terms):
    """
    Implement the Wallis product to compute an approximation of pi.

    This method calculates an approximation of pi using the Wallis 
    
    product formula.

    Parameters
    ----------
    n_terms : int
        Number of terms in the Wallis product.

    Returns
    -------
    float
        The approximation of pi of order `n_terms`.

    See Also
    --------
    https://en.wikipedia.org/wiki/Wallis_product
    """
    product = 1.0
    for i in range(1, n_terms + 1):
        product *= (4 * i ** 2) / ((4 * i ** 2) - 1)

    return product * 2
