import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The training input samples.

        y : ndarray of shape (n_samples,)
            The target values (class labels) as integers or strings.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        distances = np.linalg.norm(self.X_train_[:, np.newaxis] - X, axis=2)
        nearest_indices = np.argmin(distances, axis=0)
        y_pred = self.y_train_[nearest_indices]
        return y_pred

    def score(self, X, y):
        """Return the accuracy of the classifier on the test data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.

        y : ndarray of shape (n_samples,)
            True class labels for the test samples.

        Returns
        -------
        score : float
            The accuracy of the classifier on the test data.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
