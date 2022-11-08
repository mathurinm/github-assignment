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
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """
        Fit the object with the input data by setting X, and Y parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        Y : ndarray of shape (n_samples, )
            The output data.

        Returns
        -------
        self : Object OneNearestNeighbor
            The object itself with the data acknowledged.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """
        Predict the y values corresponding to the given X array.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, )
            The predicted data based on X.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )
        y_pred = []
        for x in X:
            minimum_dist = np.infty
            minimum_coordinate = None
            for j in range(len(self.X_)):
                dist = np.linalg.norm(x - self.X_[j])
                if dist < minimum_dist:
                    minimum_dist = dist
                    minimum_coordinate = j
            y_pred.append(self.y_[minimum_coordinate])
        return np.array(y_pred)

    def score(self, X, y):
        """
        Return the score of the prediction.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        Y : ndarray of shape (n_samples, )
            The output data.

        Returns
        -------
        score : float
            Average number of samples correctly classified
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        score = 0
        for i in range(len(y_pred)):
            if (y_pred[i] - y[i] == 0):
                score += 1
        return score / len(y_pred)
