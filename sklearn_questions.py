"""Assignment - making a sklearn estimator.

The goal of this assignment is to implement by yourself a scikit-learn
estimator for the OneNearestNeighbor and check that it is working properly.

The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`.

We also ask to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstring similar to the one in `numpy_questions`
for the methods you code and for the class. The docstring will be checked using
`pydocstyle` that you can also call at the root of the repo.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """OneNearestNeighbor classifier.

    A simple 1-nearest-neighbor classifier using Euclidean distance.
    """

    def __init__(self):  # noqa: D107
        pass

    # ----------------- compatibility helpers -----------------
    def _fit_validate_compat(self, X, y):
        """Validate X,y and set n_features_in_ (new/old sklearn)."""
        try:
            # new sklearn exposes _validate_data on estimators
            X, y = self._validate_data(X, y)
        except AttributeError:
            # old sklearn fallback
            X, y = check_X_y(X, y)
            self.n_features_in_ = X.shape[1]
        return X, y

    def _predict_validate_compat(self, X):
        """Validate X at predict time; check n_features_in_ if needed."""
        try:
            X = self._validate_data(X, reset=False)
        except AttributeError:
            X = check_array(X)
            nfi = getattr(self, "n_features_in_", None)
            if nfi is not None and X.shape[1] != nfi:
                msg = (
                    f"X has {X.shape[1]} features, but "
                    f"{self.__class__.__name__} is expecting "
                    f"{nfi} features as input"
                )
                raise ValueError(msg)

        return X
    # ---------------------------------------------------------

    def fit(self, X, y):
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : OneNearestNeighbor
            Fitted estimator.
        """
        X, y = self._fit_validate_compat(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, attributes=["X_", "y_"])
        X = self._predict_validate_compat(X)

        # squared distances between X (n,d) and self.X_ (m,d)
        diff = X[:, None, :] - self.X_[None, :, :]
        dist2 = (diff ** 2).sum(axis=2)
        nn_idx = np.argmin(dist2, axis=1)
        return self.y_[nn_idx]

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        try:
            X, y = self._validate_data(X, y, reset=False)
        except AttributeError:
            X_chk, y_chk = check_X_y(X, y)
            nfi = getattr(self, "n_features_in_", None)
            if nfi is not None and X_chk.shape[1] != nfi:
                msg = (
                    f"X has {X_chk.shape[1]} features, but "
                    f"{self.__class__.__name__} is expecting "
                    f"{nfi} features as input"
                )
                raise ValueError(msg)
            X, y = X_chk, y_chk

        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
