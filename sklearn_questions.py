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
        pass  # no parameters

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input array.

        y : ndarray of shape (n_samples,)
            True labels of X.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)  # check format X and y
        check_classification_targets(y)  # check y has classification labels
        self.classes_ = np.unique(y)  # stocking in order y values
        self.n_features_in_ = X.shape[1]  # nber columns of X

        # storing the training data
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict class labels of input samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input array whose labels need to be predicted.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)  # check fit() was called before
        X = check_array(X)  # check format of X
        # create an empty table to fill it with predictions
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        for i in range(len(X)):
            dist = np.linalg.norm(self.X_ - X[i], axis=1)  # Euclidian distance
            nearest_index = np.argmin(dist)  # index smallest distance
            y_pred[i] = self.y_[nearest_index]  # label of closest

        return y_pred

    def score(self, X, y):
        """Compute accuracy score of OneNearestNeighbor classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.

        y : ndarray of shape (n_samples,)
            True labels of test samples (X).

        Returns
        -------
        score : float
            Classification accuracy score.
        """
        self._check_n_features(X, reset=False)  # check nber features match X
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)  # make prediction of labels
        y_pred = (y_pred == y).astype(int)/len(y)  # compare prediction to true

        return y_pred.sum()
