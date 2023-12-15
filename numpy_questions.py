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
        The row and columnd index of the maximum.
    Raises
    ------
    ValueError
        If the input is not a numpy array or
        if the shape is not 2D.
    """

    if not isinstance(X, np.ndarray):
        raise ValueError("X should be a numpy array")

    if X.ndim != 2:
        raise ValueError("X should be a 2D array")

    i = 0
    j = 0

    i, j = np.unravel_index(X.flatten().argmax(), shape=X.shape)

    return i, j

    pi : float
        The approximation of order ⁠ n_terms ⁠ of pi using the Wallis product.

    unit_term = np.arange(1, n_terms + 1, 1, dtype=np.float64)
    unit_term = unit_term ** 2

    return 2 * np.prod(4*unit_term / (4 * unit_term - 1))
