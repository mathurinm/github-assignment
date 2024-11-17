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

def find_max_index(input_array):
    """Return the index of the maximum in a numpy array.

    Parameters
    ----------
    input_array : ndarray of shape (n_samples, n_features)
        The input array.

    Returns
    -------
    (row_index, col_index) : tuple(int)
        The row and column index of the maximum.

    Raises
    ------
    ValueError
        If the input is not a numpy array or
        if the shape is not 2D.
    """
    if not isinstance(input_array, np.ndarray) or input_array.ndim != 2:
        raise ValueError("Input must be a 2D numpy array")

    row_index = 0
    col_index = 0
    maximum_value = input_array[row_index, col_index]

    for row in range(input_array.shape[0]):
        for col in range(input_array.shape[1]):
            if input_array[row, col] > maximum_value:
                maximum_value = input_array[row, col]
                row_index = row
                col_index = col

    return row_index, col_index


def compute_wallis_product(num_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    num_terms : int
        Number of steps in the Wallis product. Note that `num_terms=0` will
        consider the product to be `1`.

    Returns
    -------
    approx_pi : float
        The approximation of order `num_terms` of pi using the Wallis product.
    """
    if num_terms == 0:
        return 1.0

    wallis_product_value = 1.0
    for index in range(1, num_terms + 1):
        wallis_product_value *= (4 * index ** 2) / (4 * index ** 2 - 1)

    approx_pi = 2 * wallis_product_value
    return approx_pi