import numpy as np


def max_index(X):
    """" Return the index of the maximum in a numpy array.

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
        If the input is not a numpy error or
        if the shape is not 2D.
    """
    # Value error if the data is not a numpy
    if type(X).__module__ != np.__name__:
        print("error, the data is not a numpy")
        pass
    # Value error if the shape is not 2D.
    elif X.shape != 2:
        print("error, the shape must be 2D")
        pass
    # The row and column index of the maximum
    else:
        i = 0
        j = 0
        i_count = 0
        max = 0
        for row in X:
            j_count = 0
            for val in row:
                if val > max:
                    max = val
                    i, j = i_count, j_count
                j_count += 1
            i_count += 1
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
    wallis = 1
    for i in range(n_terms + 1):
        wallis = wallis * (4 * i ** 2 / (4 * i ** 2 - 1))
    return wallis * 2