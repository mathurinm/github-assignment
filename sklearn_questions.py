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
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """One Nearest Neighbor classifier.

    This classifier predicts the class of a sample based on the class
    of its nearest neighbor in the training set, using Euclidean distance
    as the proximity metric.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels in the training data.
    n_features_in_ : int
        The number of features seen during fit.
    X_ : ndarray of shape (n_samples, n_features)
        The training input samples.
    y_ : ndarray of shape (n_samples,)
        The training target values.

    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        This method stores the training data for later use during prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Returns self to allow method chaining.

        """
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        For each sample in X, finds the nearest neighbor in the training
        set and returns its class label.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels.

        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )
        for i, x in enumerate(X):
            distances = np.sqrt(np.sum((self.X_ - x) ** 2, axis=1))
            nearest_idx = np.argmin(distances)
            y_pred[i] = self.y_[nearest_idx]
        return y_pred

    def score(self, X, y):
        """Calculate the accuracy score.

        Computes the mean accuracy of predictions on the given test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The true class labels.

        Returns
        -------
        score : float
            The mean accuracy of the classifier on the given data.

        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
