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
from sklearn.metrics import pairwise_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Train the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """
        Given a matrix of distances between test points and training points.

        Predict a label for each test point.

        Inputs:
        - X: A numpy array of shape (num_test, num_train) where X[i, j]
        gives the distance betwen the ith test
        point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,)
        containing predicted labels for the
        test data, where y[i] is the predicted label for the test point X[i].
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        # XXX fix

        dists = pairwise_distances(X, self.X_train_)
        nearest = np.argsort(dists)[:, 0]
        for i in range(len(X)):
            y_pred[i] = self.y_train_[nearest[i]]

        return y_pred

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters.
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        Returns
        -------
        accuracy : float
            Mean accuracy of self.predict(X) w.r.t. y.
        """
        X, y = check_X_y(X, y)
        # XXX fix
        accuracy = np.mean(self.predict(X) == y)
        return accuracy
