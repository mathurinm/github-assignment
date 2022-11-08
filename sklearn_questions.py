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
        """Fit the model with X_train and y_train.

        Parameters
        ----------
        -X (numpy.ndarray) : the training set of
        shape (n_samples, n_features)
        y (numpy.ndarray): the n-classes of X

        Returns
        -------
        self: the model fitted with its attributes
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predict the labels of a test sample.

        Parameters:
        ----------
        - X (numpy.ndarray): test sample

        Returns:
        ----------
        - y_pred (numpy.ndarray): an array containing the
        predicted labels for X
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        for i in range(len(X)):
            closest = np.argmin(np.sqrt(np.sum((X[i] - self.X_)**2, axis=1)))
            y_pred[i] = self.y_[closest]

        return y_pred

    def score(self, X, y):
        """Compute the average number of points correctly classified.

        Parameters:
        ----------
        - X (numpy.ndarray): test sample
        - y (numpy.ndarray): true labels of X

        Returns:
        ----------
        - accuracy (float) : the scoring metrics
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        accuracy = sum(y_pred == y) / len(y)

        return accuracy
