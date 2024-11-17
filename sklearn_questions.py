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


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    "OneNearestNeighbor classifier."

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """
        Fit the nearest neighbor classifier from the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Vérifie les données X et y
        X, y = check_X_y(X, y)

        # Vérifie que ce sont bien des cibles de classification
        check_classification_targets(y)

        # Stocke les données d’entraînement
        self.X_train_ = X
        self.y_train_ = y

        # Stocke le nombre de caractéristiques
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        # Vérifie si le modèle a été entraîné
        check_is_fitted(self)

        # Vérifie les données d’entrée
        X = check_array(X)

        # Calculer la distance euclidienne
        distances = np.linalg.norm(self.X_train_ - X[:, np.newaxis], axis=2)

        # Trouve les indices du plus proche voisin
        nearest_neighbor_idx = np.argmin(distances, axis=1)

        # Trouve la classe correspondante
        y_pred = self.y_train_[nearest_neighbor_idx]

    def score(self, X, y):
        """
        Return the accuracy of the classifier on the test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Accuracy of self.predict(X) wrt. y.
        """
        # Prédire les classes pour X
        y_pred = self.predict(X)

        # Calculer l'exactitude
        return y_pred.sum()
