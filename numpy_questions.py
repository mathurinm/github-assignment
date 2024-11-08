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
        raise ValueError("Input must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input array must be 2D.")

    max_value = X[0, 0]
    max_position = (0, 0)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] > max_value:
                max_value = X[i, j]
                max_position = (i, j)

    return max_position


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    Parameters
    ----------
    n_terms : int
        Number of steps in the Wallis product.

    Returns
    -------
    pi : float
        The approximation of order `n_terms` of pi using the Wallis product.
    """
    if n_terms == 0:
        return 1.0

    p = 1
    for k in range(1, n_terms + 1):
        num = 4 * k**2
        den = num - 1
        p *= (num / den)

    return 2 * p
