import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """
    OneNearestNeighbor classifier.

    A simple implementation of the nearest neighbor classifier which predicts
    the target of a new point as the target of the closest point in the training set.
    Proximity is measured using the Euclidean distance.

    Methods
    -------
    fit(X, y)
        Fit the model using X as training data and y as target values.
    predict(X)
        Predict the class labels for the provided data.
    score(X, y)
        Returns the mean accuracy on the given test data and labels.
    """

    def __init__(self):
        # Initialize without setting attributes
        pass

    def fit(self, X, y):
        """
        Fit the OneNearestNeighbor model to the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        # Set attributes during fit
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels for each sample in X.
        """
        check_is_fitted(self)
        X = check_array(X)

        y_pred = []
        for x in X:
            distances = np.linalg.norm(self.X_ - x, axis=1)
            nearest_index = np.argmin(distances)
            y_pred.append(self.y_[nearest_index])

        return np.array(y_pred)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data.
        y : ndarray of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of the predictions.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
