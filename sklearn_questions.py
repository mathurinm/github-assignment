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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier.

    This classifier implements the 1-Nearest Neighbor algorithm,
    which predicts the label of a query point as the label of
    the closest point in the training set using Euclidean distance.
    """

    def __init__(self):
        """Initialize the OneNearestNeighbor classifier.

        This classifier does not take any hyperparameters.
        """
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        Stores the training data and labels for future predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate the inputs
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Store the training data and labels
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Perform classification on samples in X.

        For each sample in X, predicts the label of the closest
        sample in the training data.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_queries,)
            The predicted classes.
        """
        # Check if fit has been called
        check_is_fitted(self, ["X_train_", "y_train_"])

        # Validate the input
        X = check_array(X)

        # Ensure the input has the same number of features
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Number of features in X ({X.shape[1]}) "
                f"does not match training data ({self.n_features_in_})."
            )

        # Initialize an array to store predictions
        y_pred = np.empty(X.shape[0], dtype=self.y_train_.dtype)

        # Compute predictions
        for i, x_query in enumerate(X):
            # Compute Euclidean distances to all training samples
            distances = np.linalg.norm(self.X_train_ - x_query, axis=1)
            # Find the index of the closest training sample
            nearest_idx = np.argmin(distances)
            # Assign the label of the nearest neighbor
            y_pred[i] = self.y_train_[nearest_idx]

        return y_pred

    def score(self, X, y):
        """Compute the accuracy of the classifier.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Test samples.

        y : array-like of shape (n_queries,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        # Validate the inputs
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Get predictions
        y_pred = self.predict(X)

        # Calculate accuracy
        accuracy = np.mean(y_pred == y)

        return accuracy
