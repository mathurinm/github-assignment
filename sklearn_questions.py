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
from sklearn.metrics import euclidean_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier.

    This classifier predicts the target y_k of the training sample X_k, which
    is the closest to a given point X_i. Proximity is measured with the
    Euclidean distance. The model is evaluated with accuracy, the average
    number of samples correctly classified.

    Parameters
    ----------
    None

    Attributes
    ----------
    classes_ : array-like of shape (n_classes,)
        The classes labels.
    n_features_in_ : int
        The number of features when `fit` is performed.

    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns self.

        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]  # Set the number of features

        # Store training data
        self.X_train_ = X
        self.y_train_ = y

        return self

    def predict(self, X):
        """Predict the target values for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted target values.

        """
        check_is_fitted(self)
        X = check_array(X)

        # Find the index of the nearest neighbor for each sample
        distances = euclidean_distances(X, self.X_train_)
        nearest_indices = np.argmin(distances, axis=1)

        # Predict the target values based on the nearest neighbors
        y_pred = self.y_train_[nearest_indices]

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
            Mean accuracy of self.predict(X) with respect to y.

        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        # Calculate the accuracy
        accuracy = np.mean(y_pred == y)
        return accuracy