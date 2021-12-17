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
from sklearn.metrics import euclidean_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the 1-nearest neighbors classifier from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : {array-like, sparse matrix} of shape (n_samples,)
            Target values.

        Returns
        -------
        self : KNeighborsClassifier
            The fitted k-nearest neighbors classifier.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict the label of X,

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Class labels for each data sample.
        """
        check_is_fitted(self)
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)

        return self.y_[closest]

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        score = (y == y_pred).sum() / len(y)

        return score
