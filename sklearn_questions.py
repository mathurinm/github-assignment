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
    """OneNearestNeighbor classifier using Euclidean distance.

    This classifier finds the nearest training sample for each test sample
    and assigns the label of the closest sample.

    Attributes
    ----------
    classes_ : array-like, shape (n_classes,)
        Unique class labels in the training set.

    n_features_in_ : int
        Number of features in the training data.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y
        self.n_features_in_ = X.shape[1]

        # XXX fix
        return self

    def predict(self, X):
        """Predict the class labels for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels for each sample in X.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = []

        for x in X:
            # Calculate Euclidean distances to all training points
            distances = np.sqrt(np.sum((self.X_train_ - x) ** 2, axis=1))
            # Find the nearest neighbor and its label
            nearest_neighbor_index = np.argmin(distances)
            y_pred.append(self.y_train_[nearest_neighbor_index])

        # XXX fix
        return np.array(y_pred)

    def score(self, X, y):
        """Calculate the accuracy of the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            The average number of correct predictions (accuracy).
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        # XXX fix
        score = np.mean(y_pred == y)
        return score
