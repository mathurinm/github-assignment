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


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier.

    Class to calculate One Nearest Neighbor using Euclidean distance
    as distance measure.

    Methods
    -------
    fit(self, X, y) : Fit the One nearest neighbor classifier.
    predict(self, X) : Predict class labels for the samples in X.
    score(self, X, y) : Compute the mean accuracy on the test data and labels.

    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the One nearest neighbor classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : object (an instance of itself)
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict class labels for the samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Prediction data

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
                 The predicted classes on the prediction data

        """
        check_is_fitted(self)
        X = check_array(X)

        y_pred = np.zeros(len(X), dtype=self.classes_.dtype)

        for i, sample in enumerate(X):
            distances = np.sqrt(np.sum((self.X_ - sample) ** 2, axis=1))
            nearest_neighbor_idx = np.argmin(distances)
            y_pred[i] = self.y_[nearest_neighbor_idx]

        return y_pred

    def score(self, X, y):
        """Compute the mean accuracy on the test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples

        y : array-like of shape (n_samples,)
            True labels of test samples

        Returns
        -------
        score : float
                Mean accuracy of the model on the testing data
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        return np.mean(y_pred == y)
