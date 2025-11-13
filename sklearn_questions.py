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
    """1-Nearest Neighbor (1-NN) classifier.

    Classifies samples based on the label of their single nearest
    training neighbor using Euclidean distance.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the 1-Nearest Neighbor classifier from the training dataset.

        Essentially, this method stores the training data for later use
        during prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : array-like of shape (n_samples,)

        Returns
        -------
        self : object
            Returns the fitted estimator instance.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predict class labels for the input samples.

        Computes the Euclidean distance from each sample in `X` to the
        training data and assigns the label of the nearest neighbor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)

        distances = np.sqrt(
            np.sum(
                (X[:, np.newaxis, :] - self.X_train_[np.newaxis, :, :]) ** 2,
                axis=2
            ))

        # For each test sample, get the index of the nearest training point
        nearest_indices = np.argmin(distances, axis=1)

        # Return the corresponding labels
        return self.y_train_[nearest_indices]

    def score(self, X, y):
        """Return the mean accuracy on the test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test input samples.
        y : array-like of shape (n_samples,)
            True class labels.

        Returns
        -------
        score : float
            Mean accuracy of the classifier (fraction of correct predictions).
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        return np.mean(y_pred == y)
