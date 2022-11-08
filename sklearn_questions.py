import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics import euclidean_distances

class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Nearest Neighbor Classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """
        Set the paraeters to train the classification model.
        
        X : input array (n, m)
        y : target array (n, )
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        
        return self

    def predict(self, X):
        """
        We want to access the class y for an input array X.

        X : input array (n, m)
        Y_pred : output array (n, ), prediction of classes for each X
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )
        
        for k in range(len(X)):
            distance_min = euclidean_distances(X[k], self.X_)
            nearest_index = np.argmin(distance_min)
            y_pred[k] = self.y_[nearest_index]
            
        return y_pred

    def score(self, X, y):
        """
        Compute the precision of the result ie. distance between y_pred & y.
        
        X : input array (n, m)
        y : target array (n, )
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        n = X.shape[0]
        score = 1 / n * sum(y == y_pred)
        
        return score