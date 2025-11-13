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
    Estimator that implements the 1-Nearest Neighbor algorithm to predict a new
    sample's label based on the closest training sample.

    Parameters
    ----------
     None
    """

    def __init__(self):
        """
        Init function.

        Returns
        -------
        None.

        """
        pass

    def fit(self, X, y):
        """
        Parameters
        ----------
        X: array of shape (n_samples, n_features)
        y: array of shape (n_samples,), holds the labels to predict

        Returns
        -------
        Self: itself
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
        Parameters
        ----------
        X: array of shape (n_samples, n_features)

        Returns
        -------
        y_pred: the predicted labels for each sample in X
        """
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but OneNearestNeighbor "
                f"is expecting {self.n_features_in_} features as input."
            )

        difference = X[:, None, :] - self.X_[None, :, :]
        distances = np.linalg.norm(difference, axis=2)

        closest_indx = np.argmin(distances, axis=1)
        y_pred = self.y_[closest_indx]

        return y_pred

    def score(self, X, y):
        """
        Parameters
        ----------
        X: array of shape (n_samples, n_features)
        y: array of shape (n_samples,), holds the labels to predict

        Returns
        -------
        score: float,
        the mean accuracy of the prediction against the true labels y
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        return np.mean(y_pred == y)
