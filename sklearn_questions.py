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
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,
)


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """OneNearestNeighbor classifier."""

    def __init__(self):
        """Initialize the OneNearestNeighbor classifier."""
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : ndarray of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : OneNearestNeighbor
            The fitted estimator.

        """
        X, y = validate_data(self, X, y, reset=False)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data to make predictions for.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class label for each sample.

        """
        check_is_fitted(self)

        X = validate_data(self, X, reset=False)

        diff = X[:, None, :] - self.X_[None, :, :]

        dist_matrix = np.sum(diff**2, axis=2)
        closest_indices = np.argmin(dist_matrix, axis=1)

        y_pred = self.y_[closest_indices]

        return y_pred

    def score(self, X, y):
        """Compute accuracy on the test data and labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.
        y : ndarray of shape (n_samples,)
            True labels for `X`.

        Returns
        -------
        score : float
            Accuracy of `self.predict(X)` relative to `y`.

        """
        X, y = validate_data(self, X, y, reset=False)
        y_pred = self.predict(X)

        return np.mean(y_pred == y)
