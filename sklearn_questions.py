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

    This class implements a simple one nearest neighbor classifier.
    """

    def __init__(self):  # noqa: D107
        """Initialize the OneNearestNeighbor classifier."""
        pass

    def fit(self, X, y):
        """Fit the classifier with training data.

        Args:
            X (array-like): Training data, a 2D numpy array or similar
            array-like structure.
            y (array-like): Target values, a 1D numpy array or similar
            array-like structure.

        Returns:
            self: Returns an instance of self.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Args:
            X (array-like): Test samples, a 2D numpy array or similar
            array-like structure.

        Returns:
            array: Predicted class labels for each data sample.
        """
        check_is_fitted(self)

        X = check_array(X)

        y_pred = np.zeros(X.shape[0], dtype=self.y_.dtype)

        for idx, x in enumerate(X):
            dists = np.sum((self.X_ - x) ** 2, axis=1)
            closest_idx = np.argmin(dists)
            y_pred[idx] = self.y_[closest_idx]

        return y_pred

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X (array-like): Test samples, a 2D numpy array or similar
            array-like structure.
            y (array-like): True labels for X, a 1D numpy array or similar
            array-like structure.

        Returns:
            float: Mean accuracy of self.predict(X) wrt. y.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        return np.mean(y_pred == y)
