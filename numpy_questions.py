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
    i = 0
    j = 0

    # Checking if the input is a 2D np array.
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise ValueError("Input must be a 2D numpy array")

    # Get the index of the maximum value
    max_index = np.argmax(X) # By default, the index is in the flattened array
    num_rows, num_cols = X.shape

    i = max_index // num_cols
    j = max_index % num_cols

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi."""
    if n_terms == 0:
        return 2.0

    pi = 2.0  # We initialize pi.

    for i in range(1, n_terms + 1):
        pi *= 4.0 * i**2 / (4.0 * i**2 - 1)  # Wallis product.

    return pi
