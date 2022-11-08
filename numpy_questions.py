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

X= np.arange(0, 10).reshape(-1, 1)

def max_index(X):

    i = np.argmax(np.max(X, axis=0))
    j = np.argmax(np.max(X, axis=1))

    return i, j

def wallis_product(n_terms):
    pi = 1.0

    for j in range(1, n_terms):
        pi *= 4 * j ** 2 / (4 * j ** 2 - 1)

    pi *= 2
    
    return pi
