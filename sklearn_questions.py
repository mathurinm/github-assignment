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
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier using Euclidean distance.

    The nearest neighbor classifier predicts for a point X_i the target y_k of
    the training sample X_k which is the closest to X_i.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The training input samples.

    y_ : ndarray of shape (n_samples,)
        The target values (class labels) for the training samples.

    classes_ : ndarray of shape (n_classes,)
        Array of unique class labels.

    n_features_in_ : int
        Number of features seen during `fit`.
    """

    def __init__(self):
        """Initialize the OneNearestNeighbor classifier."""
        pass

    def fit(self, X, y):
        """Fit the classifier on the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training input samples.
        y : ndarray of shape (n_samples,)
            Target values (class labels) for the training samples.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate the input and target arrays
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Store the training data and class labels
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predict the class labels for the input samples X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples to classify.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels for each input sample in X.
        """
        # Ensure the model has been fitted
        check_is_fitted(self, ["X_", "y_"])
        # Validate the input array
        X = check_array(X)

        # Initialize an array to store predictions
        y_pred = np.empty(X.shape[0], dtype=self.y_.dtype)

        # For each sample in X, find the nearest neighbor in the training data
        for i, x in enumerate(X):
            # Compute Euclidean distances from x to all samples in self.X_
            distances = np.linalg.norm(self.X_ - x, axis=1)
            # Find the index of the closest training sample
            nearest_index = np.argmin(distances)
            # Assign the label of the nearest neighbor
            y_pred[i] = self.y_[nearest_index]

        return y_pred

    def score(self, X, y):
        """Compute the accuracy of the classifier on the given test data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test input samples.

        y : ndarray of shape (n_samples,)
            True labels for the test samples.

        Returns
        -------
        score : float
            The accuracy of the classifier, defined as the proportion of
            correctly classified samples.
        """
        # Validate the input and target arrays
        X, y = check_X_y(X, y)
        # Predict the labels for the input samples
        y_pred = self.predict(X)
        # Calculate the accuracy as the mean of correctly predicted labels
        return np.mean(y_pred == y)
