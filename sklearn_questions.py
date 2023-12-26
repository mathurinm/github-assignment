import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """
    OneNearestNeighbor classifier.

    This classifier implements the 1-Nearest Neighbor algorithm for classification.
    It predicts the class label for each sample based on the nearest neighbor in
    the training set, using Euclidean distance as the proximity measure.
    """

    def __init__(self):
        """
        Initialize the OneNearestNeighbor classifier.

        This classifier does not require any hyperparameters.
        """
        pass

    def fit(self, X, y):
        """
        Fit the OneNearestNeighbor classifier model.

        This method stores the training data and corresponding labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target values (class labels).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.classes_ = np.unique(y)

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels for each sample in X.
        """
        check_is_fitted(self)
        X = check_array(X)

        distances = np.sqrt(
            ((X[:, np.newaxis, :] - self.X_[np.newaxis, :, :]) ** 2).sum(axis=2)
        )
        nearest_neighbor_indices = np.argmin(distances, axis=1)

        y_pred = self.y_[nearest_neighbor_indices]

        return y_pred

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

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
