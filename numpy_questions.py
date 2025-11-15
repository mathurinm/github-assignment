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


def max_index(X: np.ndarray) -> tuple:
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
    # 1. Contrôle du type d'entrée (Doit être un np.ndarray)
    if not isinstance(X, np.ndarray):
        raise ValueError(f"L'entrée doit être un np.ndarray, mais le type {type(X)} a été reçu.")
        
    # 2. Contrôle des dimensions (Doit être 2D)
    if X.ndim != 2:
        raise ValueError(f"Le tableau doit être 2D pour cette fonction (il a {X.ndim} dimensions).")
    # 3. Recherche de l'indice du maximum
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] == np.max(X):
                return (i, j)
    # La fonction retourne la première occurrence du maximum en parcourant les lignes puis les colonnnes.
            




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
    if not isinstance(n_terms, int):
        raise ValueError(f"Le paramètre doit être un entier,  (il est du type {type(n_terms)}).")
    if n_terms < 1:
        return 0.0
        
    product = 1.0
    
    # Itère de k=1 à n_terms (inclus)
    for k in range(1, n_terms + 1):
        # Calcule le numérateur (2k) et les dénominateurs (2k-1, 2k+1)
        # On utilise des flottants (2.0 * k) pour garantir une division flottante
        numerator = 2.0 * k
        denominator1 = numerator - 1.0
        denominator2 = numerator + 1.0
        
        # Le terme d'ordre k est (2k / (2k - 1)) * (2k / (2k + 1))
        term = (numerator / denominator1) * (numerator / denominator2)
        
        product *= term
        
    return product * 2.0  # Le produit final est multiplié par 2 pour obtenir l'approximation de pi vu que l'intégrale donne pi/2.