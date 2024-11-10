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
from sklearn.metrics.pairwise import euclidean_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier to the data.

        Parameters
        ----------
        self : the initiated non-fitted estimator
        X : ndarray of shape (n_samples, n_features)
            The training data
        Y : ndarray of shape (n_samples)
            The training labels

        Returns
        -------
        self : Fitted estimator
        """
        # Check if X and y are of correct type and shape.
        # If incorrect type, corrects, if incorrect shape, returns error.
        X, y = check_X_y(X, y)
        # Check if y is suitable for classification
        check_classification_targets(y)
        # Store the different 'unique' classes we will then need to predict
        #  based on the label array
        self.classes_ = np.unique(y)
        # Store the number of features to later know the expected
        # number of features the test data needs to have
        self.n_features_in_ = X.shape[1]
        self.X_train_ = X  # Store the training data
        self.y_train_ = y  # Store the training labels
        return self

    def predict(self, X):
        """Fit the OneNearestNeighbor classifier to the data.

        Parameters
        ----------
        self : the fitted estimator
        X : ndarray of shape (n_samples, n_features)
            The test data

        Returns
        -------
        y_pred : ndarray of shape (n_samples)
            The prediction labels
        """
        # Check if the classifier has been feeted to the data
        check_is_fitted(self)
        X = check_array(X)  # Check if X is a 2D array
        # Creates y_pred an array with the same lenght and type as X.
        # Each element of the array is the 1st class label
        # of the training labels array
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )
        # Compute the euclidian distance between the test and the training data
        distances = euclidean_distances(X, self.X_train_)
        # Find the index of the closest training data sample
        # for each test data sample
        N_N_index = np.argmin(distances, axis=1)
        # Return the label corresponding to the index of the training data
        y_pred = self.y_train_[N_N_index]

        return y_pred

    def score(self, X, y):
        """Compute the score of the OneNearestNeighbor Classifier.

        Parameters
        ----------
        self : the fitted estimator
        X : ndarray of shape (n_samples, n_features)
            The test data
        y : ndarray of shape (n_samples)
            The true labels

        Returns
        -------
        accuracy : float between 0 (very bad) and 1 (very good)
            The score of the OneNearestNeighbor classifier ie.
            the mean of the errors
        """
        # Check if X and y are of correct type and shape
        X, y = check_X_y(X, y)
        # Compute y_pred the predictions based on the trained model
        y_pred = self.predict(X)
        # Compute the score of the model i.e. how many times he is right
        # on average
        accuracy = np.mean(y_pred == y)

        return accuracy
