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


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Implement the fitting of a nearest neighbor classifier.

        Parameters
        ----------
        X : numpy.darray of shape (number of samples, number of features)
            The input array we want to fit on.

        y : numpy.darray of shape (number of samples)
            The target of the samples present in X.

        Returns
        -------
        self : OneNearestNeighbor
            Fitted estimator.

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
        """Implement the prediction of a target y according to the entry X.

        Parameters
        ----------
        X : numpy.darray of shape (number of samples, number of features)
            The input array we want to predict.

        Returns
        -------
        y : numpy.darray of shape (number of samples)
            The prediction of the samples in X according to the
            nearest neighbor classifier.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        # XXX fix

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has {} features, but {} is expecting {} features as input"
                .format(
                    X.shape[1],
                    self.__class__.__name__,
                    self.n_features_in_,
                )
            )

        diffs = X[:, np.newaxis, :] - self.X_[np.newaxis, :, :]
        dist_sq = np.sum(diffs ** 2, axis=2)

        nearest_indices = np.argmin(dist_sq, axis=1)

        y_pred = self.y_[nearest_indices]

        return y_pred

    def score(self, X, y):
        """Give the score of a prediction.

        Parameters
        ----------
        X : numpy.darray of shape (number of samples, number of features)
            The samples we test.

        y : numpy.darray of shape (number of samples)
            The true labels of the X we try to predict with the classifier.
        Returns
        -------
        score : float
            The mean accuracy of the predictor on the given data and target.

        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        # XXX fix
        y_pred = (y_pred == y).astype(float) / len(y)

        return y_pred.sum()
