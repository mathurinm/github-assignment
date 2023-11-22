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
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import _check_sample_weight


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y, sample_weight=None):
        """Store X and y values.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array containing the observations of the features.

        y : array of shape (n_samples, )
            The input array containing the actual values.

        Returns
        -------
        self : object
            Returns self.

        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        if sample_weight is not None:
            # Handle sample weights if provided
            sample_weight = _check_sample_weight(sample_weight, X)
        self.sample_weight_ = sample_weight
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predicts the target values for input data X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array containing the observations of the features.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted target values for X.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )
        distances = euclidean_distances(X, self.X_)

        closest_indices = np.argmin(distances, axis=1)

        y_pred = self.y_[closest_indices]

        return y_pred

    def score(self, X, y):
        """Scores the model based on accuracy.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array containing the observations of the features.

        y : array of shape (n_samples, )
            The true target values.

        Returns
        -------
        score : float
            The accuracy of the model on the provided data and labels.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        score = accuracy_score(y, y_pred)
        return score
