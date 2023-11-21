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

    It classifies instances.

    Based on the closest training
    example in the feature space.

    """

    def __init__(self):  # noqa: D107
        """Initialize our classifier."""
        pass

    def fit(self, X, y):
        """Fit the 1-Nearest Neighbor classifier model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        # Store the classes seen during fit
        self.n_features_in_ = X.shape[1]
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Class labels for each data sample.
        """
        # Check if fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)

        # Calculate distances beetween X and self.X
        distances = np.sqrt(((self.X_[np.newaxis, :, :] -
                              X[:, np.newaxis, :]) ** 2).sum(axis=2))
        # Predict the class of the nearest neighbor
        y_pred = self.y_[distances.argmin(axis=1)]
        return y_pred

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        # Return the accuracy score
        n = y.shape[0]
        score = np.sum(y_pred == y)/n
        return score
