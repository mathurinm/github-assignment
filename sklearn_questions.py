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
    """
    OneNearestNeighbor classifier.

    This classifier implements the One Nearest Neighbor algorithm which predicts
    the class of a given sample based on the class of the closest sample in the
    training data. The proximity is measured using the Euclidean distance.

    Methods:
    fit(X, y):
        Fits the model using the training data.
    predict(X):
        Predicts the class labels for the provided data.
    score(X, y):
        Returns the mean accuracy on the given test data and labels.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """
        Fits the data to the self instance.

        Parameters
        ----------
        X: np.ndarray
            2D array containing the features of the observations,
        y: np.ndarray
            1D array containing the classes of the observations.

        Returns:
        ----------
        self (OneNearestNeighbor): the fitted instance.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.X_ = X
        self.y_ = y
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predicts the class of new observations based on the training data.

        Parameters:
        X (np.ndarray):
            2D array containing the features of the observations.

        Returns:
        y_pred (np.ndarray):
            1D array containing the predicted class based on the training data.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0], dtype=self.classes_.dtype
        )
        for i, x in enumerate(X):
            min_dist = np.inf
            for j, x_train in enumerate(self.X_):
                dist = np.linalg.norm(x - x_train)
                if dist < min_dist:
                    min_dist = dist
                    y_pred[i] = self.y_[j]

        return y_pred

    def score(self, X, y):
        """
        Scores the prediction of the OneNearestNeighbor model for given data.
        Compares predicted classes with real classes and returns mean accuracy.

        Parameters:
        X (np.ndarray): 2D array containing the features of the observations,
        y (np.ndarray): 1D array containing the real classes of the observations.

        Returns:
        y_pred.sum() (int): the score of the model.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        y_pred = (y_pred == y).astype(float) / len(y)
        return y_pred.sum()
