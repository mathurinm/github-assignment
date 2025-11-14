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
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array
from sklearn.metrics import pairwise_distances_argmin_min


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier. 
        
        This function stores training data X and the labels y.

        Parameters
        ----------
        X : Training data (n_samples, n_features).

        y : Target labels (n_samples).

        Returns
        -------
        self : returns the fitted classifier.

        Raises
        ------
        ValueError
            If X and y have different numbers of samples
        """
        
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y)
        return self

    def predict(self, X):
        """Returns the predicted class for a data set in an numpy array.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.

        Returns
        -------
        y_pred : ndarray of shape (n_samples)
            The predicted classes for the n_samples.
        """

        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        indexes, _ = pairwise_distances_argmin_min(X, self.X_)
        y_pred = self.y_[indexes]
        return y_pred

    def score(self, X, y):
        """Returns the score of the OneNearestNeighbor on a data set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.
        y : ndarray of shape (n_samples)
            The true classes of the samples.

        Returns
        -------
        score : float
            The percentage of samples accurately predicted.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        score = np.mean(y_pred == y)
        return score
