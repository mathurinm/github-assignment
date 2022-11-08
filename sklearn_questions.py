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
        """
        X : input array (n, m)
        y : target array (n, )
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.m = X.shape[1]
        self.n = X.shape[0]

        self.X_ = X
        self.y_ = y
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

        # y_pred.sum()
        score = 1 / n * sum(y == y_pred)
        
        return score

#pydocstyle