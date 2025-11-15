"""Assignment - making a sklearn estimator."""

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
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
        X = check_array(X, ensure_2d=True)
        y = check_array(y, ensure_2d=False)
        check_classification_targets(y)

        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predict the label of each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples for which to predict labels.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels.
        """
        check_is_fitted(self)
        X = check_array(X, ensure_2d=True)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input"
            )

        distance = np.linalg.norm(self.X_[None, :, :] - X[:, None, :], axis=2)
        nearest_index = np.argmin(distance, axis=1)

        return self.y_[nearest_index]

    def score(self, X, y):
        """Compute the accuracy of the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples used to evaluate the model.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            The accuracy of the predictions, between 0 and 1.
        """
        check_is_fitted(self)
        X = check_array(X, ensure_2d=True)
        y = check_array(y, ensure_2d=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input"
            )
        y_pred = self.predict(X)

        return (y_pred == y).mean()
