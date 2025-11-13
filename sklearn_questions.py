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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit a nearest neighbor model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The features vector.
        y : ndarray of shape (n_samples, 1)
            The target vector.

        Returns
        -------
        The fitted model.
        """
        X, y = check_X_y(X, y)
        X = check_array(X)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict target from a feature vector with a nearest neighbor model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The feature vector from which to predict y.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, 1)
            The predicted value for y.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0], dtype=self.classes_.dtype
        )

        distances = np.linalg.norm(
            X[:, np.newaxis, :] - self.X_[np.newaxis, :, :], axis=2
        )
        nearest_idx = np.argmin(distances, axis=1)
        y_pred = self.y_[nearest_idx]

        return y_pred

    def score(self, X, y):
        """Return the score of a model evaluating against ground truth.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The features vector.
        y : ndarray of shape (n_samples, 1)
            The target vector.

        Returns
        -------
        score : float
            The score of the model.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        score = np.mean(y_pred == y)
        return score
