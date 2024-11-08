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
from sklearn.metrics.pairwise import euclidean_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the model using the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if X is None or y is None:
            raise ValueError("X and y must not be None.")

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.X_train_ = X
        self.y_train_ = y

        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict on.
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels for each sample.
        """
        check_is_fitted(self, ['X_train_', 'y_train_'])
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("Error in the number of features of the model.")

        distances = euclidean_distances(X, self.X_train_)
        nearest_neighbor_indices = np.argmin(distances, axis=1)
        y_pred = self.y_train_[nearest_neighbor_indices]

        return y_pred

    def score(self, X, y):
        """Compute the accuracy of the model on the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The test data.
        y : array-like of shape (n_samples,)
            The true target values (class labels).
        Returns
        -------
        score : float
            The accuracy of the model on the test data.
        """
        if X is None or y is None:
            raise ValueError("X and y must not be None.")

        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)

        return accuracy
