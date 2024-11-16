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
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier.

    Implementation of the nearest neighbor classifier with proximity measured
    using the Euclidean distance.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Array of unique class labels in the training data.

    n_features_in_ : int
        Number of features in the training data.

    X_ : ndarray of shape (n_samples, n_features)
        Training input samples.

    y_ : ndarray of shape (n_samples,)
        Training target values.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor model using the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The training input samples.
        y : ndarray of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : OneNearestNeighbor
            The fitted classifier.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        distances = np.linalg.norm(self.X_[:, np.newaxis] - X, axis=2)
        nearest_indices = np.argmin(distances, axis=0)
        y_pred = self.y_[nearest_indices]
        return y_pred

    def score(self, X, y):
        """Calculate the accuracy of the classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The test input samples.
        y : ndarray of shape (n_samples,)
            The true class labels for X.

        Returns
        -------
        accuracy : float
            The accuracy of the classifier as the proportion of correctly
            classified samples.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)

        return accuracy
