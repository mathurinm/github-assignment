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
  
    if not isinstance(X, np.ndarray): # check if it is a Numpy array 
        raise ValueError("The input is not a Numpy array")
    
    if X.ndim != 2: # check if the array is 2D
        raise ValueError("Input array must be 2D.")

    max_index = np.unravel_index(np.argmax(X), X.shape) # finds the index of maximum value
    
    return max_index


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product
"""
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
   
    p = 1 # Initilization

    for n in range(1, n_terms+1): 
        p *= 4 * n ** 2 / (4 * n ** 2 - 1) # Implement the Wallis product, here p = pi/2

    return 2*p # Have to multiply p by 2 to get pi
