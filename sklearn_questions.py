"""Module for implementing a custom scikit-learn estimator.

This module contains the implementation of the OneNearestNeighbor
classifier, which predicts the label of a query point based on the
closest point in the training set using Euclidean distance.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier.

    A nearest neighbor classifier that predicts the label of a query point
    based on the closest point in the training set using Euclidean distance.
    """

    def __init__(self):
        """Initialize the OneNearestNeighbor classifier."""
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor model using the training data.

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
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X  # Store training data
        self.y_ = y  # Store target values
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Predict the target for each query point in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted target values for X.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.empty(X.shape[0], dtype=self.y_.dtype)

        for i, x in enumerate(X):
            distances = np.linalg.norm(self.X_ - x, axis=1)
            nearest_index = np.argmin(distances)
            y_pred[i] = self.y_[nearest_index]

        return y_pred

    def score(self, X, y):
        """Return the accuracy of the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        score : float
            Mean accuracy of the classifier on the test data.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
