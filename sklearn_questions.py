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
from sklearn.utils.validation import (
    check_X_y,
    check_is_fitted,
    validate_data,
)
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """One-nearest neighbor classifier.

    This classifier implements the 1-nearest neighbor rule using the
    Euclidean distance to find, for each sample, the closest point in the
    training set and predict its class label.
    """

    def __init__(self):  # noqa: D107
        # No hyper-parameters for this simple estimator.
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
        # This sets n_features_in_ and handles validation in a
        # sklearn-compatible way.
        X, y = validate_data(self, X, y, accept_sparse=False)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples for which to predict class labels.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)

        # Use reset=False so sklearn checks consistency with n_features_in_
        X = validate_data(self, X, reset=False)

        # Compute pairwise Euclidean distances between X and training data
        diff = X[:, np.newaxis, :] - self.X_[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)

        # For each sample in X, find index of nearest neighbor in training data
        nearest_idx = np.argmin(distances, axis=1)

        # Predict the label of the nearest neighbor
        y_pred = self.y_[nearest_idx]

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
            Mean accuracy of the predictions on X with respect to y.
        """
        # We still validate (X, y) as a proper supervised dataset.
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        return float(np.mean(y_pred == y))
