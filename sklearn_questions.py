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
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the classifier by memorizing the training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training input samples.

        y : ndarray of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : OneNearestNeighbor
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict class labels for samples in `X`.

        Parameters
        ----------
        X : ndarray of shape (n_queries, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_queries,)
            Predicted class labels.
        """
        check_is_fitted(
            self, attributes=["X_", "y_", "classes_", "n_features_in_"]
            )
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input"
            )

        diff = X[:, np.newaxis, :] - self.X_[np.newaxis, :, :]
        dists = np.sum(diff ** 2, axis=2)
        nn_idx = np.argmin(dists, axis=1)
        return self.y_[nn_idx]

    def score(self, X, y):
        """
        Compute accuracy of the classifier on the given test data and labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.

        y : ndarray of shape (n_samples,)
            True labels for `X`.

        Returns
        -------
        accuracy : float
            Mean accuracy of predictions on `X` compared to `y`.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
