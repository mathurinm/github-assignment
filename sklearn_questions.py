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
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import validate_data


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """OneNearestNeighbor classifier.

    This classifier predicts the class of a sample based on the class of
    its nearest neighbor in the training set, using Euclidean distance.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        Store the training data to use for predictions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values (class labels).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        # Store training data
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        For each sample, find the nearest neighbor in the training set
        and return its class label.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        # For each test sample, find the nearest training sample
        for i, x_test in enumerate(X):
            # Compute Euclidean distances to all training samples
            distances = np.sqrt(np.sum((self.X_ - x_test) ** 2, axis=1))
            # Find the index of the nearest neighbor
            nearest_idx = np.argmin(distances)
            # Predict the class of the nearest neighbor
            y_pred[i] = self.y_[nearest_idx]

        return y_pred

    def score(self, X, y):
        """Calculate the accuracy score.

        Compute the mean accuracy of predictions on the given test data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.
        y : ndarray of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of predictions.
        """
        X, y = validate_data(self, X, y, reset=False)
        y_pred = self.predict(X)

        # Calculate accuracy: proportion of correct predictions
        accuracy = np.mean(y_pred == y)

        return accuracy
