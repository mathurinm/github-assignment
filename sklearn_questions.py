"""
Assignment - creating a scikit-learn estimator.

This module implements a OneNearestNeighbor classifier that follows the
scikit-learn API, including fit, predict, and score methods.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (
    check_X_y,
    check_is_fitted,
    validate_data,
)
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """One-Nearest-Neighbor classifier.

    This classifier assigns to each test sample the label of the closest
    training sample according to the Euclidean distance.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training input samples.

        y : ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : OneNearestNeighbor
            Fitted estimator.
        """
        # validate_data will:
        # - check shapes and types
        # - set self.n_features_in_
        X, y = validate_data(self, X, y)
        check_classification_targets(y)

        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels for each input sample.
        """
        check_is_fitted(self)

        # reset=False -> do not overwrite n_features_in_,
        # but check consistency with what was seen in fit.
        X = validate_data(self, X, reset=False)

        # Pairwise Euclidean distances: shape (n_test, n_train)
        distances = np.linalg.norm(
            X[:, np.newaxis, :] - self.X_[np.newaxis, :, :],
            axis=2,
        )

        # Index of closest training point for each test sample
        nn_idx = np.argmin(distances, axis=1)

        return self.y_[nn_idx]

    def score(self, X, y):
        """Compute the accuracy of predictions on X compared to y.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test input samples.

        y : ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Accuracy = proportion of correctly classified samples.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
