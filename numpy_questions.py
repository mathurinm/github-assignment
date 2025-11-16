# """Assignment - using numpy and making a PR.

# The goals of this assignment are:
#     * Use numpy in practice with two easy exercises.
#     * Use automated tools to validate the code (`pytest` and `flake8`)
#     * Submit a Pull-Request on github to practice `git`.

# The two functions below are skeleton functions. The docstrings explain what
# are the inputs, the outputs and the expected error. Fill the function to
# complete the assignment. The code should be able to pass the test that we
# wrote. To run the tests, use `pytest test_numpy_questions.py` at the root of
# the repo. It should say that 2 tests ran with success.

# We also ask to respect the pep8 convention: https://pep8.org.
# This will be enforced with `flake8`. You can check that there is no flake8
# errors by calling `flake8` at the root of the repo.
# """
import numpy as np


def max_index(X):
    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    # Flatten index of the max
    flat_index = np.argmax(X)

    # Convert to 2D index
    i, j = np.unravel_index(flat_index, X.shape)

    return i, j


def wallis_product(n_terms):
    if n_terms < 0:
        raise ValueError("n_terms must be non-negative.")

    if n_terms == 0:
        return 1.0

    n = np.arange(1, n_terms + 1)
    terms = (4 * n * n) / (4 * n * n - 1)

    product = np.prod(terms)

    return 2 * product

