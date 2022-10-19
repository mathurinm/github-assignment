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
    """"
    ValueError
        If the input is not a numpy error or
        if the shape is not 2D.
    """  
    if type(X).__module__ != np.__name__: 
        print("error, the data is not a numpy")
        pass
    elif X.shape != 2: 
        print("error, the shape must be 2D")
        pass
    else:
    #  The row and columnd index of the maximum.
        i = np.argmax(X, axis=0)
        j = np.argmax(X, axis=1)

    return i, j


def wallis_product(n_terms):
    wallis = 1
    for i in range(n_terms -1) +1:
        wallis = wallis * (4*(i**2)/(4*(i**2) - 1))
    return wallis*2