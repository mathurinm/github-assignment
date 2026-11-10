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
from sklearn.utils.validation import validate_data
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """OneNearestNeighbor classifier.

    Predict for each sample the target of the closest training sample,
    using the Euclidean distance.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels seen during `fit`.
    n_features_in_ : int
        The number of features seen during `fit`.
    X_ : ndarray of shape (n_samples, n_features)
        The training samples stored during `fit`.
    y_ : ndarray of shape (n_samples,)
        The training targets stored during `fit`.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The training samples.
        y : ndarray of shape (n_samples,)
            The target labels of the training samples.

        Returns
        -------
        self : OneNearestNeighbor
            The fitted classifier.

        Raises
        ------
        ValueError
            If `X` and `y` have inconsistent lengths, or if `y` is not a
            valid classification target.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # XXX fix
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict the class labels for the provided samples.

        Each sample is assigned the target of its nearest training sample
        in Euclidean distance.

        Parameters
        ----------
        X : ndarray of shape (n_queries, n_features)
            The samples to classify.

        Returns
        -------
        y_pred : ndarray of shape (n_queries,)
            The predicted class label for each sample.

        Raises
        ------
        NotFittedError
            If the classifier has not been fitted yet.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        # XXX fix
        X = validate_data(self, X, reset=False)
        dist = X[:, None, :] - self.X_[None, :, :]
        norm = np.linalg.norm(dist, axis=-1)
        idx = np.argsort(norm, axis=-1)[:, 0]
        y_pred = self.y_[idx]

        return y_pred

    def score(self, X, y):
        """Return the accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray of shape (n_queries, n_features)
            The test samples.
        y : ndarray of shape (n_queries,)
            The true labels of the test samples.

        Returns
        -------
        score : float
            The fraction of samples correctly classified.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        # XXX fix
        return np.mean(y_pred == y)
