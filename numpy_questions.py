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
    (i, j) : tuple(int)
        The row and column index of the maximum.

    Raises
    ------
    ValueError
        If the input is not a numpy array or
        if the shape is not 2D.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise ValueError("Input must be a 2D numpy array")

    max_value_index = np.unravel_index(np.argmax(X, axis=None), X.shape)
    return max_value_index


def wallis_product(n_terms):
    """
    Implement the Wallis product to compute an approximation of pi.

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
        return 2.0

    pi_over_two = 1.0
    for i in range(1, n_terms + 1):
        pi_over_two *= (4.0 * i ** 2) / ((4.0 * i ** 2) - 1)

    return pi_over_two * 2
