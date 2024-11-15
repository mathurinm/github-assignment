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

    A nearest neighbor classifier that predicts for a point X_i the target y_k
    of the training sample X_k which is the closest to X_i. The proximity is
    measured using the Euclidean distance.
    """

    def __init__(self):
        """Empty"""
        pass

    def fit(self, X, y):
        """Fit the nearest neighbor classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input training data.

        y : ndarray of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate inputs
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Store training data
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Predict using the nearest neighbor classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels.
        """
        # Ensure the estimator is fitted
        check_is_fitted(self)

        # Validate the input
        X = check_array(X)

        # Predict based on the nearest neighbor
        y_pred = np.array([
            self.y_[np.argmin(np.linalg.norm(self.X_ - x, axis=1))]
            for x in X
        ])
        return y_pred

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input test data.

        y : ndarray of shape (n_samples,)
            The true labels.

        Returns
        -------
        accuracy : float
            The mean accuracy.
        """
        # Validate inputs
        X, y = check_X_y(X, y)

        # Predict and compute accuracy
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
