"""
OneNearestNeighbor sklearn assignment.

Implementation of a simple nearest-neighbor classifier.
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

    This estimator assigns to each input sample the label of the closest
    training point based on Euclidean distance.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : array-like of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : OneNearestNeighbor
            The fitted estimator.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predict class labels for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels.
        """
        check_is_fitted(self, ["X_", "y_", "n_features_in_"])
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model expects "
                f"{self.n_features_in_}."
            )

        diff = X[:, None, :] - self.X_[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        nearest_idx = np.argmin(distances, axis=1)

        return self.y_[nearest_idx]

    def score(self, X, y):
        """Return the mean accuracy on the given test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Accuracy of the classifier.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
