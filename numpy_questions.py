"""Assignment - using numpy and making a PR.

The goals of this assignment are:
    * Use numpy in practice with two easy exercises.
    * Use automated tools to validate the code (`pytest` and `flake8`)
    * Submit a Pull-Request on github to practice `git`.

The two functions below are skeleton functions. The docstrings explain what
are the inputs, the outputs and the expected error. Fill the function to
complete the assignment. The code should be able to pass the test that we
wrote. To run the tests, use `pytest test_numpy_question.py` at the root of
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

    # TODO
    # check if x is a numpy array
    if not isinstance(X, np.ndarray):
        raise ValueError("Input is not a NumPy array.")

    # check if shape of X is 2D
    if X.ndim != 2:
        raise ValueError("Input array is not 2D.")

    # find the index of maximum
    max_index = np.unravel_index(np.argmax(X, axis=None), X.shape)

    # update i and j
    i += max_index[0]
    j += max_index[1]

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
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    # check if input is non-negative
    if not isinstance(n_terms, int) or n_terms < 0:
        raise ValueError("Number of terms must be a non-negative integer.")

    # the original product is 1
    product = 1.0

    # calculate the wallis product
    for i in range(1, n_terms + 1):
        numerator = 2 * i
        denominator = 2 * i
        product *= (numerator ** 2) / (denominator ** 2 - 1)

    # calculate the pi approximation
    pi_approximation = 2 * product

    return pi_approximation
