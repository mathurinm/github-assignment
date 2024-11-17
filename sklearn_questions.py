import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):
        """Initialize OneNearestNeighbor classifier."""
        pass

    def fit(self, X, y):
        """
        Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            The input training data.
        y : np.ndarray, shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : OneNearestNeighbor
            The fitted classifier.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self._X_train = np.array(X)
        self._y_train = np.array(y)
        return self

    def predict(self, X):
        """
        Predict the class labels for the input samples.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            The input samples for which to predict labels.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted class labels for each input sample.
        """
        check_is_fitted(self)
        X = check_array(X)

        # Compute the distance between each element of X and _X_train
        dist = np.linalg.norm(X[:, np.newaxis] - self._X_train, axis=2)

        # Find the nearest neighbor
        nearest_indices = np.argmin(dist, axis=1).astype(int)
        y_pred = self._y_train[nearest_indices]
        return y_pred

    def score(self, X, y):
        """
        Compute the accuracy of the classifier.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            The input samples for testing.
        y : np.ndarray, shape (n_samples,)
            True labels for the input samples.

        Returns
        -------
        result : float
            The result of the predictions compared to the true labels.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        result = np.mean(y_pred == y)
        return result
