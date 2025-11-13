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
    """
    OneNearestNeighbor Classifier.

    This estimator implements the 1-Nearest Neighbor classification algorithm.
    It predicts the label of a new sample based on the label of the single
    closest training sample (nearest neighbor) using Euclidean distance.

    Parameters
    ----------
    No parameters are needed for this simple implementation.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.n_features_in_ = X.shape[1]
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """
        Predict the class label for the provided data.


        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels for the input samples.
        """
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but OneNearestNeighbor "
                f"is expecting {self.n_features_in_} features as input."
            )

        diff = X[:, None, :] - self.X_[None, :, :]
        distances = np.linalg.norm(diff, axis=2)

        nearest_idx = np.argmin(distances, axis=1)
        y_pred = self.y_[nearest_idx]

        return y_pred

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) vs. y.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        return np.mean(y_pred == y)
