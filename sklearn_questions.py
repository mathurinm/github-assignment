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
        """Initialize the model."""
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor model on training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data (input)
        y : ndarray of shape (n_samples,)
            Class labels of training data

        Returns
        -------
        self : object
            The fitted estimator
        """
        X, y = check_X_y(X, y)  # check if X/y are of expected type/shape
        check_classification_targets(y)  # check if y is an array of discrete
        self.classes_ = np.unique(y)  # store all possible labels
        self.n_features_in_ = X.shape[1]  # store number of features
        # store training data and labels
        self.X_train_ = X
        self.y_train_ = y

        return self  # return current instance of the OneNearestNeighbor class

    def predict(self, X):
        """Predict class labels (based on test data).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        -- The input data to classify

         Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels for each sample
        """
        check_is_fitted(self)  # ensure the model has been fitted
        X = check_array(X)  # validate input data
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype)  # initialize prediction array

        # For each sample in X, find the closest training sample
        for i, x_test in enumerate(X):
            squared_diff = (self.X_train_ - x_test) ** 2
            distances = np.sqrt(np.sum(squared_diff, axis=1))
            closest_index = np.argmin(distances)  # closest sample index
            y_pred[i] = self.y_train_[closest_index]  # predict the class

        return y_pred

    def score(self, X, y):
        """Calculate accuracy of model (average number of times it's right).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        -- The input on which to score the model
        y : ndarray of shape (n_samples,)
        -- The true labels for the input

        Returns
        -------
        score : float
        Average number of times the model correctly classifies samples.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        return np.mean(y_pred == y)
