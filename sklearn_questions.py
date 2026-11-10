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
        """Memorizes the examples in X and their classes in y

        Variables:
        X -> features of the training data
        y -> classes of the training data

        Returns:
        self-> fitted One-Nearest-Neighbor model
        """

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        # XXX fix

        return self

    def predict(self, X):
        """Predicts the class of the given X,
        from the class of the Nearest Neighbor in the training set

        Variables:
        X -> features table from which we want to predict the class
        self -> fitted model

        Returns:
        y_pred -> the predicted classes for X
        """
        check_is_fitted(self, ["X_train_", "y_train_"])
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError

        y_pred = np.empty(len(X), dtype=self.y_train_.dtype)

        for i, x in enumerate(X):
            distances = np.sum((self.X_train_ - x) ** 2, axis=1)
            nearest_index = np.argmin(distances)

            y_pred[i] = self.y_train_[nearest_index]

        return y_pred
        # XXX fix
        return y_pred

    def score(self, X, y):
        """Calculates the accuracy (i.e. proportion of right answers)
        of the model given data X and labels y

        Variables:
        X -> features table from which we want to predict the class
        y -> true classes of the data in X
        self -> fitted model

        Returns:
        accuracy -> the accuracy of the model on the given data X and labels y
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        # XXX fix
        return np.mean(y_pred == y)
