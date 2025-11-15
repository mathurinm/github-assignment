"""Assignment - making a sklearn estimator.

The goal of this assignment is to implement by yourself a scikit-learn
estimator for the OneNearestNeighbor and check that it is working properly.

The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`.

We also ask to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstring similar to the one in `numpy_questions`
for the methods you code and for the class. The docstring will be checked
using
`pydocstyle` that you can also call at the root of the repo.
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Le but du fit est juste de stocker les données d'entraînement.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Les échantillons d'entraînement.
        y : array, shape (n_samples,)
            Les étiquettes cibles des échantillons d'entraînement.

        Returns
        -------
        self : object
            L'estimateur entraîné (c'est à dire contenant les données
            d'entraînement).
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # Rajout des données d'entraînement dans l'estimateur
        self.X_train_ = X
        self.y_train_ = y

        # Marquer l'estimateur comme "entraîné"
        self.is_fitted_ = True

        return self

    def predict(self, X):
        """ Le but de la méthode predict est de prédire les étiquettes pour les
        échantillons de test X en utilisant les données d'entraînement qui
        ont précédemment été stockées dans l'estimateur lors de l'appel à fit.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Les échantillons de test pour lesquels prédire les étiquettes.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Les étiquettes de classe prédites pour chaque échantillon de test.
        """
        # Vérification que l'estimateur a bien été entraîné
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        # Itération sur chaque échantillon de test
        for i in range(X.shape[0]):
            x_test = X[i, :]

            # Calcul de la distance euclidienne entre x_test et tous
            # les points
            # d'entraînement (self.X_)

            distances = np.sum((self.X_train_ - x_test)**2, axis=1)

            # Trouver l'indice du point d'entraînement le plus proche
            closest_index = np.argmin(distances)

            # L'étiquette prédite est celle du voisin le plus proche
            y_pred[i] = self.y_train_[closest_index]

        return y_pred

    def score(self, X, y):
        """ Retourne l'exactitude moyenne (accuracy) des prédictions sur X
        par rapport à y.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Les échantillons de test.
        y : array, shape (n_samples,)
            Les étiquettes cibles vraies pour X.

        Returns
        -------
        score : float
            Le pourcentage d'étiquettes bien pédites.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        return np.mean(y_pred == y)
