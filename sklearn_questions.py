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
    """OneNearestNeighbot classifier."""

    def _init_(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fitting the One nearest neighbout classifier"""

        # Checking that X and y have a correct shape
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Storing the classes and number of features
        self.classes_ = np.unique(y)
        self.X_ = X  # Store the training data
        self.y_ = y  # Store the target labels
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predicting target values"""

        # Checking if the model has been fitted
        check_is_fitted(self)

        # Checking the input array and initializing predictions
        X = check_array(X)
        y_pred = np.empty(len(X), dtype=self.y_.dtype)

        for i, x in enumerate(X):

            # Calculating Euclidean distances between x and all samples
            distances = np.linalg.norm(self.X_ - x, axis=1)

            # Finding the index of the nearest neighbor
            nearest_neighbor_idx = np.argmin(distances)

            # Assigning the label of the nearest neighbor to y_pred[i]
            y_pred[i] = self.y_[nearest_neighbor_idx]

        return y_pred

    def score(self, X, y):
        """Return mean accuracy"""

        X, y = check_X_y(X, y)

        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)

        return accuracy
