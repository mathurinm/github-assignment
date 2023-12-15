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
    i = 0
    j = 0

    # Checking if the input is a 2D np array.
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise ValueError("Input must be a 2D numpy array")

    # Get the index of the maximum value
    max_index = np.argmax(X)  # By default, the index is in the flattened array
    num_rows, num_cols = X.shape

    i = max_index // num_cols
    j = max_index % num_cols

    return i, j

    # terms in the product. For example 10000.

    if n_terms == 0:
        return 2.0

    pi = 2.0  # We initialize pi.

    for i in range(1, n_terms + 1):
        pi *= 4.0 * i**2 / (4.0 * i**2 - 1)  # Wallis product.
        
    return pi
