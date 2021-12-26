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
    "OneNearestNeighbor classifier."

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Write docstring.

        And describe parameters
        """

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y

        # XXX fix
        return self

    def predict(self, X):
        """Write docstring.

        And describe parameters
        """
        # Compute all pairwise distances between X and self.X_
        distances = sklearn.metrics.pairwise.pairwise_distances(X, Y=self.X_, metric='euclidean' )
        # Get indices to sort them
        indices_sorted = np.argsort(distances, axis=1)
        #  Get indices of neighbors
        indices_neighbors = indices_sorted[:, :1]
        # Get labels of neighbors
        Y_neighbors = self.y_[indices_neighbors]
        #  Find the predicted labels y for each entry in X
        mode , k = stats.mode(Y_neighbors, axis=1)
        y_pred = np.array(mode.T, dtype=int)

        # XXX fix
        return y_pred

    def score(self, X, y):
        """Write docstring.

        And describe parameters
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        l = len(y)
        sum = 0
        for i in range(l):
            if y[i] == y_pred[0,i]:
                sum = sum + 1
        score = sum / l
        # XXX fix
        return score
