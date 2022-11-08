import numpy as np


def max_index(X):
    """ Return the index of the maximum in a numpy array.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input array.

    Returns
    -------
    (i, j) : tuple(int)
        The row and columnd index of the max.

    Raises
    ------
    ValueError
        If the input is not a numpy error or
        if the shape is not 2D.
    """
    # Value error if the data is not a numpy
    if type(X).__module__ != np.__name__:
        print("error, the data is not a numpy")
        raise ValueError
    # Value error if the shape is not 2D.
    elif X.shape != 2:
        print("error, the shape must be 2D")
        raise ValueError
    # The row and column index of the maximum
    else:
        global i, j
        i, j = np.unravel_index(X.argmax(), X.shape)
    return i, j


def wallis_product(n_terms):
    """ Implement the Wallis product to compute an approximation of pi.
        
    Parameters
    ----------
    n_terms : int
        Number of steps in the Wallis product.
        Note that `n_terms=0` will consider the product to be `1`.
    Returns
    -------
    pi : float
        The approximation of order `n_terms` of pi using the Wallis product.
    """
    if(n_terms == 0):
        pi = 1
    
    else:
        wallis = 2
        for k in range(n_terms+1):
            wallis = wallis * (4 * k**2 / (4 * k**2 - 1))
        pi = wallis * 2
    return pi