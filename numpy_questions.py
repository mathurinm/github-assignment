"""Assignment - using numpy and making a PR.

The goals of this assignment are:
    * Use numpy in practice with two easy exercises.
    * Use automated tools to validate the code (pytest and flake8)
    * Submit a Pull-Request on github to practice git.

The two functions below are skeleton functions. The docstrings explain what
are the inputs, the outputs and the expected error. Fill the function to
complete the assignment. The code should be able to pass the test that we
wrote. To run the tests, use pytest test_numpy_questions.py at the root of
the repo. It should say that 2 tests ran with success.

We also ask to respect the pep8 convention: https://pep8.org.
This will be enforced with flake8. You can check that there is no flake8
errors by calling flake8 at the root of the repo.
"""
import numpy as np


def max_index(X):
    """
    Return the index of the maximum value in a numpy array.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input array.

    Returns
    -------
    (i, j) : tuple of int
        Row and column indices of the maximum value in X.

    Raises
    ------
    ValueError
        If X is not a numpy array or is not 2D.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a numpy array.")  # Validate input
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array.")  # Ensure array is 2D
    return np.unravel_index(np.argmax(X), X.shape)


def wallis_product(n_terms):
    """
    Compute an approximation of pi using the Wallis product.

    Parameters
    ----------
    n_terms : int
        Number of terms to compute in the Wallis product.

    Returns
    -------
    pi : float
        Approximation of pi using n_terms.

    Raises
    ------
    ValueError
        If n_terms is less than 0.
    """
    if n_terms < 0:
        raise ValueError("Number of terms cannot be negative.")
    if n_terms == 0:
        return 1.0  # Handle the case where n_terms is 0.

    product = 1.0
    for i in range(1, n_terms + 1):
        product *= (4 * i**2) / ((4 * i**2) - 1)

    return product * 2








