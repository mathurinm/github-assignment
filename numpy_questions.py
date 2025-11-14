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
    # Check that X is a 2D numpy array
    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input array must be 2D.")

    max_value = X.max()

    # All positions where X == max_value
    positions = np.argwhere(X == max_value)

    # We only keep the first one (row, column)
    row, col = positions[0]

    return int(row), int(col)


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
    # Basic validation on n_terms
    if not isinstance(n_terms, int):
        raise ValueError("n_terms must be an integer.")
    if n_terms < 0:
        raise ValueError("n_terms must be non-negative.")
    if n_terms == 0:
        return 1.0

    # Vectorized computation of the Wallis product
    n = np.arange(1, n_terms + 1, dtype=float)
    left = 2.0 * n / (2.0 * n - 1.0)
    right = 2.0 * n / (2.0 * n + 1.0)
    product = np.prod(left * right)

    return float(2.0 * product)
