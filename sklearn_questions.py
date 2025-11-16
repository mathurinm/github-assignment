"""
Assignment - creating a scikit-learn estimator.

This module implements a OneNearestNeighbor classifier that follows the
scikit-learn API, including fit, predict, and score methods.
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
        """Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training input samples.
        y : ndarray of shape (n_samples,)
            Target labels associated with each training sample.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # XXX fix
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict class labels for given samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples for which to predict labels.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class label for each sample.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        # XXX fix
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but OneNearestNeighbor "
                f"is expecting {self.n_features_in_} features as input"
            )

        diff = X[:, np.newaxis, :] - self.X_[np.newaxis, :, :]
        distances = np.sum(diff ** 2, axis=2)

        nearest_idx = np.argmin(distances, axis=1)
        y_pred[:] = self.y_[nearest_idx]

        return y_pred

    def score(self, X, y):
        """Compute accuracy of the classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.
        y : ndarray of shape (n_samples,)
            True target labels.

        Returns
        -------
        score : float
            Accuracy of predictions: fraction of correctly classified samples.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        # XXX fix
        return np.mean(y_pred == y)
