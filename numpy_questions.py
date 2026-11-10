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
import math


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
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")

    elif X.size == 0:
        raise ValueError("X must be 2D")

    elif len(X.shape) != 2:
        raise ValueError("X must be 2D")

    else:

        i = 0
        j = 0

        max = 0
        n_samples, n_features = X.shape
        for s in range(n_samples):
            for f in range(n_features):
                if X[s][f] > max:
                    max = X[s][f]
                    i = s
                    j = f

        return (i, j)


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
    if n_terms == 0:
        return 1

    else:
        L = []

        for i in range(1, n_terms+1):
            temp = (2*i/(2*i-1))*(2*i/(2*i+1))
            L.append(temp)

        # Multiplies all the elements in the list
        pi = 2 * math.prod(L)

        return pi
