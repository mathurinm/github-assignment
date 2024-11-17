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

    This classifier implements a One nearest neighbor algorithm using
    Euclidean distance. It predicts the label of a point as the label of
    the closest training sample.
    """

    def __init__(self):
        """Initialize the OneNearestNeighbor classifier."""
        pass

    def fit(self, X, y):
        """
        Fit the classifier using the training data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training samples (inpiut).
        y : array of shape (n_samples,)
            These target values (class labels) for the training input.

        Returns
        -------
        self : object
            Returns the instance itself.
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
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class label for each input sample.
        """
        check_is_fitted(self)
        X = check_array(X)

        y_pred = []
        for x in X:
            # Compute Euclidean distances
            distances = np.linalg.norm(self.X_ - x, axis=1)
            # Find the index of the closest training sample
            nearest_index = np.argmin(distances)
            # Append the corresponding label our predictions (y_pred)
            y_pred.append(self.y_[nearest_index])

        return np.array(y_pred)

    def score(self, X, y):
        """
        Compute the averag accuracy of the classifier.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The test input samples.
        y : array-like of shape (n_samples,)
            The true labels for the test input.

        Returns
        -------
        score : float
            The accuracy of the classifier, defined as the proportion
            of correctly classified samples.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
