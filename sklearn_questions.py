"""Assignment - making a sklearn estimator.

The goal of this assignment is to implement by yourself a scikit-learn
estimator for the OneNearestNeighbor and check that it is working properly.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """One nearest neighbor classifier."""

    def __init__(self):  # noqa: D107
        # no hyperparameters
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.

        y : array-like of shape (n_samples,)
            Training class labels.

        Returns
        -------
        self : OneNearestNeighbor
            The fitted classifier.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predict class labels for the provided samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, attributes=["X_", "y_"])
        X = check_array(X)

        # Enforce consistency of number of features with training data
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input"
            )

        # Compute Euclidean distances to all training points
        # X shape: (n_test, n_features)
        # self.X_ shape: (n_train, n_features)
        distances = np.linalg.norm(
            self.X_[np.newaxis, :, :] - X[:, np.newaxis, :],
            axis=2,
        )
        nearest_idx = np.argmin(distances, axis=1)
        y_pred = self.y_[nearest_idx]

        return y_pred

    def score(self, X, y):
        """Return the accuracy of the classifier on the given test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True class labels.

        Returns
        -------
        score : float
            Classification accuracy.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
