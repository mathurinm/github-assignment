import numpy as np


def max_index(X):
    
    if not isinstance(X, np.ndarray):
        raise ValueError("X is not a numpy array")
    
    if len(X.shape) != 2:
        raise ValueError("X is not a two-dimensional array")

    flatIndex = np.argmax(X)
    size = X.shape
    maxIndices = np.unravel_index(flatIndex, size)

    return maxIndices

def wallis_product(n_terms):

    if n_terms == 0:
        return 2.0

    result = 1.0
    for k in range(1, n_terms + 1):
        term = (2 * k) / (2 * k - 1) * (2 * k) / (2 * k + 1)
        result *= term

    # Multiply by 2 to get the final approximation of pi
    return 2 * result



X = np.random.randint(0, 20, size = (5,8))
result = max_index(X)
print(X, result)

n_terms = (10000)
approximation = wallis_product(n_terms)
print("Approximation of pi:", approximation)