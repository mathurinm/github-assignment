"""Custom One Nearest Neighbor classifier using sklearn interface."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """One Nearest Neighbor classifier.

    Predicts the label of a sample as the label of the closest training point
    using Euclidean distance.
    """

    def __init__(self):
        """Initialize the classifier (no parameters)."""
        pass

    def fit(self, X, y):
        """Fit the classifier on training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : OneNearestNeighbor
            Fitted classifier.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """Predict labels for input samples X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        distances = np.linalg.norm(X[:, None, :] - self.X_train_[None, :, :], axis=2)
        nearest_idx = np.argmin(distances, axis=1)
        return self.y_train_[nearest_idx]

    def score(self, X, y):
        """Compute accuracy of the classifier on test data."""
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
