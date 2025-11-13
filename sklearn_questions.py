"""
Implementation of a One-Nearest Neighbor classifier.

Implementation of a One-Nearest Neighbor classifier adhering to
the scikit-learn estimator interface.

"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """One-Nearest-Neighbor classifier.

    This classifier predicts the class of a sample by finding the
    closest sample in the training data (using Euclidean distance)
    and assigning its class.
    """

    def __init__(self):  # noqa: D107
        """Initialize the OneNearestNeighbor classifier."""
        pass

    def fit(self, X, y):
        """Fit the One-Nearest-Neighbor classifier.

        This method stores the training data (X and y) to be used
        during prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict the class labels for provided data.

        For each sample in X, finds the closest training sample
        (using Euclidean distance) and returns its label.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)

        y_pred = np.empty(shape=len(X), dtype=self.y_.dtype)

        for i, x_test in enumerate(X):
            sq_distances = np.sum((self.X_ - x_test)**2, axis=1)
            nearest_index = np.argmin(sq_distances)
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
            Mean accuracy of self.predict(X) wrt. y.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        return np.mean(y_pred == y)
