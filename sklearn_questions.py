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
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        """Initialize class."""
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        Args :
            X, np array : Training data.
            y, np array: Target values.

        Returns :
            self : OneNearestNeighbor
            The fitted OneNearestNeighbor classifier.

        Errors :
            ValueError : if X and y don't have corresponding shapes.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Predict the class of a new data.

        Args :
            X : np array with features and samples.
        Returns :
            The closest point, thus predicting the class since\
            it is a 1NN model.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self.y_train_[np.argmin(
            euclidean_distances(X, self.X_train_), axis=1)]

    def score(self, X, y):
        """Return the average the number of samples correctly classified."""
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        return accuracy_score(y, y_pred)
