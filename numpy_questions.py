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

from typing import Tuple

import numpy as np


def check_input_format(X) -> None:
    """Check the correctness of the input array.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input array.

    Returns
    -------
    None.
    """
    if type(X) != np.ndarray or len(X.shape) != 2:
        raise ValueError


def max_index(X: np.ndarray) -> Tuple[int, int]:
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
        If the input is not a numpy error or
        if the shape is not 2D.
    """
    # Check input correctness.
    check_input_format(X)

    # Find 2d-index of the max element.
    height, width = X.shape
    argmax = np.argmax(X.ravel())

    i = argmax // height
    j = argmax % width

    return i, j


def wallis_product(n_terms: int) -> float:
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
    # Start from 2 as Wallis product gives approximation of pi / 2.
    pi_approx = 2.

    if n_terms == 0:
        return pi_approx

    for curr_term in range(1, n_terms + 1):
        pi_approx *= (4 * curr_term ** 2) / (4 * curr_term ** 2 - 1)

    return pi_approx
