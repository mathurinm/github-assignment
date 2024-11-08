"""Assignment - using numpy and making a PR.

The goals of this assignment are:
    * Use numpy in practice with two easy exercises.
    * Use automated tools to validate the code (`pytest` and `flake8`)
    * Submit a Pull-Request on github to practice `git`.

The two functions below are skeleton functions. The docstrings explain what
are the inputs, the outputs and the expected error. Fill the function to
complete the assignment. The code should be able to pass the test that we
wrote. To run the tests, use `pytest test_numpy_questions.py` at the root of
the repo. It should say that 2 tests ran with success.

We also ask to respect the pep8 convention: https://pep8.org.
This will be enforced with `flake8`. You can check that there is no flake8
errors by calling `flake8` at the root of the repo.
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
        The row and columnd index of the maximum.

    Raises
    ------
    ValueError
        If the input is not a numpy array or
        if the shape is not 2D.
    """
    i = 0
    j = 0

    # Check if X is a numpy array
    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    # Check if X is 2D
    if X.ndim != 2:
        raise ValueError("Input array must be 2D.")
    
    # Find the index of the maximum value in the flattened array
    index_flat = np.argmax(X)
    
    # Convert the flattened index to 2D (row, column)
    i, j = np.unravel_index(index_flat, X.shape)

    return i, j


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
    product = 1.0
    
    # Loop over the number of terms
    for n in range(1, n_terms + 1):
        # Wallis product term: (4n^2) / (4n^2 - 1)
        product *= (4 * n * n) / (4 * n * n - 1)
    
    # Multiply by 2 to approximate pi
    pi_approx = 2 * product
    return pi_approx