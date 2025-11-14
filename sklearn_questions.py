import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        distances = np.linalg.norm(
            X[:, None, :] - self.X_train_[None, :, :],
            axis=2)
        nearest_idx = np.argmin(distances, axis=1)
        return self.y_train_[nearest_idx]

    def score(self, X, y):
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
