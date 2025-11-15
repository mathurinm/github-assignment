"""Assignment - making a sklearn estimator."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predict labels for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify. Must have same number of features
            as training data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels for each input sample.
        """
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input."
            )

        distances = np.linalg.norm(self.X_[None, :, :] - X[:, None, :], axis=2)
        nearest_idx = np.argmin(distances, axis=1)

        return self.y_[nearest_idx]

    def score(self, X, y):
        """Compute the accuracy of the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples used to evaluate the model.
        y : array-like of shape (n_samples,)
            True labels corresponding to X.

        Returns
        -------
        score : float
            Accuracy of predictions, between 0 and 1.
        """
        check_is_fitted(self)
        X, y = check_X_y(X, y)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input."
            )

        y_pred = self.predict(X)
        return np.mean(y_pred == y)
