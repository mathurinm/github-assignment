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
        Fit the OneNearestNeighbor classifier to the data.

        Parameters
        ----------
        X : array-like or Pandas.DataFrame
            Input data of shape (n_sample, n_features).
        y : array_like or Pandas.Series
            Target values associated to X of shape (n_samples, ).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.X_train_ = X
        self.y_train_ = y

        return self

    def predict(self, X):
        """
        Return the label prediction for given data, using the ONN classifier.

        Parameters
        ----------
        X : array-like or Pandas.DataFrame
            New data for which to predict the label, \
            of shape (n_sample, n_features).

        Returns
        -------
        y_pred : array-like or Pandas.Series
            Label prediction vector of shape (n_new, ).
        """
        check_is_fitted(self)
        X = check_array(X)

        distances = np.linalg.norm(X[:, None] - self.X_train_, axis=2)
        argmins = np.argmin(distances, axis=1)
        y_pred = self.y_train_[argmins]
        return y_pred

    def score(self, X, y):
        """
        Return the average number of samples correctly \
        predicted by the classifier.

        Parameters
        ----------
        X : array-like or Pandas.DataFrame
            Input data to predict, of shape (n_sample, n_features).
        y : array_like or Pandas.Series
            Actual target values associated to X of shape (n_sample, ).

        Returns
        -------
        score : float
            Number of correct guesses divided by n_sample.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        score = (y_pred == y).mean()
        return score
