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
for the methods you code and for the class. The docstring will be checked using
`pydocstyle` that you can also call at the root of the repo.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the model. But for KNN (Here 1NN) there is no real fit to do.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
                The input array.

        y : ndarray of shape (n_samples,)
                The output array.

        Returns
        -------
        self
        """
        X, y = validate_data(self, X, y)

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.X_train_ = X
        self.y_train_ = y

        # XXX fix
        return self

    def predict(self, X):
        """Return the class predicted.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
                    The output array.

        Raises
        ------
        ValueError
            if the shape is not 2D.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        if len(X.shape) != 2:
            raise ValueError

        X = check_array(X)
        # if self.n_features_in_ != X.shape[1]:
        #     raise ValueError

        y_pred = np.full(
            shape=len(X),
            fill_value=self.classes_[0],
            dtype=self.classes_.dtype,
        )

        # pour chaque ligne (point) de X
        #   calculer la distance entre le point et tous les autres points.
        #   garder le point avec la plus petit distance
        #   retourner sa classe

        # le faire si possible avec le broadcasting

        # la distance avec le broadcasting mais que pour le premier point
        np.sqrt(np.sum((self.X_train_ - X[0]) ** 2, axis=1))

        # On veut un tableau (n_test,n_train,p) le p est le même car on a le
        # même nombre de features

        # On a un tableau ou la distance au carré entre le point i et j est
        # à la case (i,j)
        # J'ai enlevé le sqrt puisque que pas utile pour les comparaisons
        distances = np.sum(
            (self.X_train_[None, :, :] - X[:, None, :]) ** 2, axis=2
        )

        # Maintenant on veut trouver pour chaque ligne le point
        # avec la plus petite distance
        closest_points = np.argmin(distances, axis=1)
        # Ça retourne pour chaque ligne l'indice de la plus petite distance
        y_pred = self.y_train_[
            closest_points
        ]  # et ça ça doit être les classes de ces points.

        return y_pred

    def score(self, X, y):
        """Return the score.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.
        y : ndarray of shape (n_samples,)
            The output array.

        Returns
        -------
        score : float,
            the accuracy of the model between 0 and 1.

        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        score = np.sum(y_pred == y) / len(y)

        return score
