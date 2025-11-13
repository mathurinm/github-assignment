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
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets

# Compatibility shim for scikit-learn versions without `validate_data`
try:  # pragma: no cover - import behavior depends on sklearn version
    from sklearn.utils.validation import validate_data as _sk_validate_data
except Exception:  # pragma: no cover
    _sk_validate_data = None


def _validate_data(estimator, X, y=None, reset=True):
    """Validate X and optional y with sklearn or a local fallback.

    - If sklearn provides `validate_data`, delegate to it.
    - Otherwise, use `check_X_y` / `check_array` and ensure `n_features_in_`
      is set during fit (reset=True) and enforced during predict/score
      (reset=False) with the standard error message format expected by
      estimator checks.
    """
    if _sk_validate_data is not None:
        if y is None:
            return _sk_validate_data(estimator, X, reset=reset)
        return _sk_validate_data(estimator, X, y, reset=reset)

    # Fallback for older scikit-learn versions
    if y is None:
        X_checked = check_array(X)
        if reset:
            estimator.n_features_in_ = X_checked.shape[1]
        else:
            if (
                hasattr(estimator, "n_features_in_")
                and X_checked.shape[1] != estimator.n_features_in_
            ):
                raise ValueError(
                    f"X has {X_checked.shape[1]} features, but "
                    f"{estimator.__class__.__name__} is expecting "
                    f"{estimator.n_features_in_} features as input"
                )
        return X_checked
    else:
        X_checked, y_checked = check_X_y(X, y)
        if reset:
            estimator.n_features_in_ = X_checked.shape[1]
        else:
            if (
                hasattr(estimator, "n_features_in_")
                and X_checked.shape[1] != estimator.n_features_in_
            ):
                raise ValueError(
                    f"X has {X_checked.shape[1]} features, but "
                    f"{estimator.__class__.__name__} is expecting "
                    f"{estimator.n_features_in_} features as input"
                )
        return X_checked, y_checked


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """One-nearest neighbor classifier.

    This estimator assigns to each input sample the target of the closest
    training sample using the Euclidean distance.

    The classifier exposes `classes_` and `n_features_in_` after fitting and
    follows the scikit-learn estimator API.
    """

    def __init__(self):  # noqa: D107
        pass

    # No custom tags to maximize cross-version compatibility

    def fit(self, X, y):
        """Fit the classifier on the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : OneNearestNeighbor
            Fitted estimator.
        """
        if y is None:
            # Be tolerant for older/newer sklearn checks that may call
            # fit(X, None) when requires_y tag is not enforced.
            X = _validate_data(self, X, reset=True)
            self.X_ = X
            self.y_ = None
            self.n_features_in_ = X.shape[1]
            return self
        X, y = _validate_data(self, X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        # store training set for nearest neighbor lookup
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = _validate_data(self, X, reset=False)
        # Compute pairwise squared Euclidean distances efficiently
        A = np.sum(self.X_ ** 2, axis=1)[None, :]  # shape (1, n_train)
        B = np.sum(X ** 2, axis=1)[:, None]        # shape (n_test, 1)
        C = X @ self.X_.T                          # shape (n_test, n_train)
        d2 = A + B - 2 * C
        nn_index = np.argmin(d2, axis=1)
        return self.y_[nn_index]

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.
        y : ndarray of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of predictions on X with respect to y.
        """
        X, y = _validate_data(self, X, y, reset=False)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
