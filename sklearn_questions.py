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

    This classifier implements the 1-nearest neighbor algorithm, which predicts
    the class of a new sample based on the class of its nearest neighbor in
    the training set using Euclidean distance.

    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """
        Fit the model from data in X and y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # store the training data
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X {X.shape[1]} features, but OneNearestNeighbor"
                             f"is expecting {self.n_features_in_} features.")

        # Initialize predictions array
        y_pred = np.zeros(X.shape[0], dtype=self.classes_.dtype)

        # For each sample in X
        for i, sample in enumerate(X):
            # Calculate distances to all training samples
            distances = np.sqrt(np.sum((self.X_ - sample) ** 2, axis=1))
            # Find index of nearest neighbor
            nearest_neighbor_idx = np.argmin(distances)
            # Predict the class of the nearest neighbor
            y_pred[i] = self.y_[nearest_neighbor_idx]

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

        return np.mean(y_pred == y)
