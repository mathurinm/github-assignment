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
    """Return the index of the maximum in a numpy array."""
    if not isinstance(X, np.ndarray):
        raise ValueError(" X must be a numpy array.")
    if X.ndim != 2
        raise ValueError("X must be 2D.")

    i, j = np.unravel_index(np.argmax(X), X.shape)
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi."""
    if n_terms == 0:
        return 2.0

    n = np.arange(1, n_terms + 1)
    terms = (2.0 * n / (2.0 * n - 1)) * (2.0 * n / (2.0 * n + 1))
    pi_approximation = np.prod(terms)

    return pi_approximation * 2

