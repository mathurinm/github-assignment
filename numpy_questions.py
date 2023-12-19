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
        The row and column index of the maximum.

    Raises
    ------
    ValueError
        If the input is not a numpy array or
        if the shape is not 2D.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise ValueError("Input must be a 2D numpy array")

    index = np.argmax(X)
    i, j = np.unravel_index(index, X.shape)
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    Parameters
    ----------
    n_terms : int
        Number of steps in the Wallis product.

    Returns
    -------
    pi : float
        The approximation of pi using the Wallis product.
    """
    product = 1.0
    for i in range(1, n_terms + 1):
        product *= (4 * i ** 2) / ((4 * i ** 2) - 1)
    return product * 2


# Example usage
if __name__ == "__main__":
    # Example for max_index
    X = np.array([[1, 2, 3], [4, 5, 6]])
    print("Max index:", max_index(X))

    # Example for wallis_product
    print("Approximation of pi:", wallis_product(10000))
