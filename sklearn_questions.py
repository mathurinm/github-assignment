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
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fitting the model: store the complete training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features_in_)
        The input array. Each row represents an observaion
        of the different parmateres considered (columns).

        y: array of shape (n_samples)
        A vector with the class of each observation.

        Returns
        -------
        y_pred : array of shape (n_samples)
        A vector of predictions whose length is the number of test cases.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        # XXX fix
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def calculate_euclidean(self, a, b):
        """Euclidean distance between a point & data."""
        return np.sqrt(np.sum((a - b) ** 2))

    def predict(self, X):
        """Prediction of class label for every row in the data set X.

        Parameters
        ----------
        X : ndarray of shape (n_tests, n_features_in_)
        The input array.

        Returns
        -------
        y_pred : array of shape (n_tests)
        A vector of predictions whose length
        is the number of test cases.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )
        for i in range(len(X)):
            test_sample = X[i, :]
            euc_distances = [self.calculate_euclidean(test_sample,
                                                      b) for b in self.X_]
            sorted_k = np.argsort(euc_distances)[:1]
            nearest_neighb = [self.y_[y] for y in sorted_k]
            prediction = stats.mode(nearest_neighb)[0][0]
            y_pred[i] = prediction
        return y_pred

    def score(self, X, y):
        """Return the score of prediction made compared with real value y.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features_in_)
        The input array. Each row represents an observaion
        of the different parmateres considered (columns)

        y: array of shape (n_samples)
        A vector with the class of each observation.

        Returns
        -------
        accurancy : int
        The fraction of predictions our model got right.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        # XXX fix
        accuracy = sum(y_pred == y) / len(y)
        return accuracy
