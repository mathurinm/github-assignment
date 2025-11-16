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

from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

# Try importing validate_data from newer sklearn
try:
    from sklearn.utils.validation import validate_data
except ImportError:
    # Fallback validate_data for older sklearn versions (used on CI)
    def validate_data(estimator, X, y=None, **kwargs):
        """
        Fallback implementation of validate_data for older sklearn versions.

        Parameters
        ----------
        estimator : estimator instance
            The estimator calling this function.
        X : array-like
            Input data.
        y : array-like, optional
            Target values.
        kwargs : dict
            Additional arguments, such as dtype or ensure_2d.

        Returns
        -------
        X : ndarray
            Validated input data.
        y : ndarray, optional
            Validated target values when provided.
        """
        if y is not None:
            X, y = check_X_y(X, y, **kwargs)
            estimator.n_features_in_ = X.shape[1]
            return X, y
        else:
            X_checked = check_array(X, **kwargs)
            if kwargs.get("reset") is False:
                if X_checked.shape[1] != estimator.n_features_in_:
                    raise ValueError(
                        f"X has {X_checked.shape[1]} features, but "
                        f"{estimator.__class__.__name__} was fitted with "
                        f"{estimator.n_features_in_} features."
                    )
            return X_checked


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier."""
        # Validate input (numeric only)
        X, y = check_X_y(X, y, dtype="numeric")

        check_classification_targets(y)

        # Store training data
        self.X_ = X
        self.y_ = y

        # Required sklearn attribute
        self.classes_ = np.unique(y)

        # Required for compatibility with predict
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predict class labels using 1-nearest neighbor."""
        check_is_fitted(self)

        # Validate input and check feature consistency
        X = validate_data(
            self, X,
            dtype="numeric",
            ensure_2d=True,
            reset=False
        )

        n_test = X.shape[0]
        y_pred = np.empty(n_test, dtype=self.y_.dtype)

        for i in range(n_test):
            distances = np.sum((self.X_ - X[i]) ** 2, axis=1)
            nn_index = np.argmin(distances)
            y_pred[i] = self.y_[nn_index]

        return y_pred

    def score(self, X, y):
        """Return mean accuracy."""
        X, y = check_X_y(X, y, dtype="numeric")
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
