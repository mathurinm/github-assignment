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
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """This class fits an OneNearestNeighbor model."""

    def __init__(self):
        """Initiate the class object."""
        pass

    def fit(self, X, y):
        """Fits the model based on the inputted data.

        Args:
            self : The class object
            X : The covariates of the data
            y : The class or target variables of the data

        Returns:
            self : The class object, now fitted.

        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, xs):
        """Predicts the classes of a vector of observations.

        Args:
            self : The class object (who must be already fitted)
            xs : an array of observations to predict

        Returns:
            np.array : the predicted array of classes
        """
        check_is_fitted(self)
        distances = pairwise_distances(self.X_, xs)
        indexes = np.argsort(distances, axis=0)[0, :]
        return self.y_[indexes]

    def score(self, X, y):
        """Calculate the accuracy of the model.

        Args:
            self : The class object
            X : The covariates of our data
            y : the class variable of our data

        Returns :
            float : The accuracy of our model
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        test = sum(x == y for x, y in zip(y_pred, y))
        return test/len(y)
