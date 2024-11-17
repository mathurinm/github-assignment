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

    This classifier predicts the label of a sample based on the closest
    training sample using the Euclidean distance.

    Attributes
    ----------
    classes_ : array-like of shape (n_classes,)
        The unique class labels.

    n_features_in_ : int
        The number of features in the training data.

    X_ : array-like of shape (n_samples, n_features)
        The training data.

    y_ : array-like of shape (n_samples,)
        The labels for the training data.
    """

    def __init__(self):
        """
        Initialize the OneNearestNeighbor classifier.

        This method sets up the object without requiring any additional
        parameters. It does not perform any actions during initialization.
        """
        pass

    def fit(self, X, y):
        """
        Fit the OneNearestNeighbor classifier.

        Stores the training data and labels for later use in prediction.

        Parameters
        ----------
        X : array-like shape (n_samples, n_features)
            Training data.

        y : array-like shape (n_samples,)
            Labels for training data.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Validate the input and output arrays
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Store training data and labels
        self.X_ = X
        self.y_ = y

        # Store unique class labels and number of features
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """
        Predict the class labels for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels for the test data.
        """
        # Ensure the model is fitted and validate the input
        check_is_fitted(self)
        X = check_array(X)

        # Compute predictions
        y_pred = []
        for x in X:
            # Compute Euclidean distances to all training samples
            distances = np.linalg.norm(self.X_ - x, axis=1)
            # Find the index of the closest sample
            nearest_index = np.argmin(distances)
            # Predict the label of the closest sample
            y_pred.append(self.y_[nearest_index])

        return np.array(y_pred)

    def score(self, X, y):
        """
        Compute the accuracy of the classifier.

        The accuracy is the number of correctly predicted samples
        divided by the total number of samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        y : array-like of shape (n_samples,)
            True labels for the test data.

        Returns
        -------
        score : float
            The accuracy of the classifier.
        """
        # Validate the input and output arrays
        X, y = check_X_y(X, y)
        # Predict the labels for the test data
        y_pred = self.predict(X)
        # Compute and return the accuracy
        return np.mean(y_pred == y)
