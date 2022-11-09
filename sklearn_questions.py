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
        """Return the class object initialized.

        Parameters
        ----------
        self : the class object

        X : n-dimmensional array of shape (n, m)
            The input array.

        y : n-dimmensional array of targets
            The input array

        Returns
        -------
        self : the class object initialized
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Return the y_pred.

        Parameters
        ----------
        self : the class object

        X : n-dimmensional array of shape (n, m)
            The input array.

        Returns
        -------
        y_pred : n-dimmensional array of shape (n, m)
            An array with the predicted targets of X.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )
        n = X.shape[0]
        for k in range(n):
            difference = X[k] - self.X_
            euclidian_distance = np.linalg.norm(difference, axis=1)
            m = range(len(euclidian_distance))
            i = m[euclidian_distance.tolist().index(min(euclidian_distance))]
            y_pred[k] = self.y_[i]

        return y_pred

    def score(self, X, y):
        """Return the accuracy score of prediction.

        Parameters
        ----------
        self : the class object

        X : n-dimmensional array of shape (n, m)
            The input array.

        y : n-dimmensional array of shape (n, m)
            The input array.

        Returns
        -------
        final_score : floating point number
            The accuracy score of prediction.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        score = []
        n = y_pred.shape[0]
        for k in range(n):
            if y_pred[k] == y[k]:
                score.append(1)
        final_score = sum(score) / n
        return final_score
