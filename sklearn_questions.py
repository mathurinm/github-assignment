"""Assignment - making a sklearn estimator.

The goal of this assignment is to implement by yourself a scikit-learn
estimator for the OneNearestNeighbor and check that it is working properly.

The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples correctly classified). You need to implement the `fit`,
`predict`, and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`.

We also ask to respect the pep8 convention: https://pep8.org.
This will be enforced with `flake8`. You can check that there are no flake8
errors by calling `flake8` at the root of the repo.

Finally, you need to write docstring similar to the one in `numpy_questions`
for the methods you code and for the class. The docstring will be checked
using `pydocstyle` that you can also call at the root of the repo.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """
    OneNearestNeighbor classifier.

    Implements a nearest neighbor classifier that predicts the class
    of the closest point in the training set using Euclidean distance.
    """

    def __init__(self):
        """
        Initialize the OneNearestNeighbor classifier.

        This constructor is intentionally left blank as no hyperparameters
        are required.
        """
        pass

    def fit(self, X, y):
        """
        Fit the nearest neighbor classifier from the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted classifier.
        """
        X, y = check_X_y(X, y)  # Validate input
        check_classification_targets(y)  # Ensure classification targets
        self.X_train_ = X  # Store training data
        self.y_train_ = y  # Store target values
        self.n_features_in_ = X.shape[1]  # Store number of features
        self.classes_ = np.unique(y)  # Store unique target classes
        return self

    def predict(self, X):
        """
        Predict the class labels for the provided test data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)  # Ensure model is fitted
        X = check_array(X)  # Validate test data

        # Compute distances between test points and training points
        distances = np.linalg.norm(
            self.X_train_ - X[:, np.newaxis], axis=2
        )
        nearest_neighbor_idx = np.argmin(distances, axis=1)
        return self.y_train_[nearest_neighbor_idx]

    def score(self, X, y):
        """
        Return the accuracy of the classifier on the test data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data.

        y : ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Accuracy of self.predict(X) compared to y.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)  # Compute accuracy


