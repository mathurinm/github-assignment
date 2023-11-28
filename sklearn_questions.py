import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.multiclass import unique_labels

class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """
    OneNearestNeighbor classifier.
    A simple implementation of the nearest neighbor classifier which predicts
    the target of a new point as the target of the closest point in the training set.
    Proximity is measured using the Euclidean distance.
    """

    def fit(self, X, y):
        """
        Fit the OneNearestNeighbor model to the training data.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        check_classification_targets(y)
        self.X_ = X
        self.y_ = y
        self.classes_ = unique_labels(y)
        return self

    def predict(self, X):
        """
        Predict the class labels for the provided data.
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
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
