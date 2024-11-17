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
    """
    OneNearestNeighbor classifier.

    This classifier implements a simple nearest neighbor classification
    algorithm. For a given sample, it finds the closest point in the training
    set (using Euclidean distance) and assigns the label of that point.
    """

    def __init__(self):
        """Initialize the OneNearestNeighbor classifier."""
        pass

    def fit(self, X, y):
        """
        Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input training data.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to classify.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels for each sample.
        """
        check_is_fitted(self)

        X = check_array(X)

        y_pred = []
        for x in X:
            distances = np.linalg.norm(self.X_ - x, axis=1)
            nearest_index = np.argmin(distances)
            y_pred.append(self.y_[nearest_index])
        return np.array(y_pred)

    def score(self, X, y):
        """
        Return the accuracy of the classifier on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        The input test data.
        y : array-like of shape (n_samples,)
        The true class labels.
        Returns
        -------
        accuracy : float
        Accuracy score, defined as the ratio of correctly classified
        samples to total samples.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
