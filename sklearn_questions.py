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
    """One-nearest-neighbor classifier.

    This classifier predicts the label of a sample using the label of the
    closest training sample according to the Euclidean distance.
    """

    def __init__(self):  # noqa: D107
        # Pas d'hyperparamètres pour ce modèle
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : OneNearestNeighbor
            Fitted estimator.
        """
        # Vérifications standard scikit-learn
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Attributs nécessaires pour scikit-learn
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # On mémorise les données d'entraînement
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict class labels for the given test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels for each sample in X.
        """
        check_is_fitted(self)
        X = check_array(X)

        # Vérifier que le nombre de features correspond
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Number of features of X does not match training")

        n_samples_test = X.shape[0]
        y_pred = np.empty(n_samples_test, dtype=self.y_.dtype)

        # Pour chaque point de test, on cherche le point d'entraînement
        # le plus proche (distance euclidienne) et on copie son label.
        for i in range(n_samples_test):
            # différences entre x_i et tous les X_ de train
            diffs = self.X_ - X[i]
            # distances euclidiennes (norme L2)
            dists = np.linalg.norm(diffs, axis=1)
            # index du plus proche voisin
            nearest_index = np.argmin(dists)
            y_pred[i] = self.y_[nearest_index]

        return y_pred

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of predictions on X compared to y.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        # proportion de bonnes prédictions
        return np.mean(y_pred == y)

