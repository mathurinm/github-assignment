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
    """OneNearestNeighbor classifier.

    A classifier that predicts the class of tested points by finding the
    closest training points in Euclidean space.

    Has the following attributes:
    ----------
    classes_ : array, shape (n_classes,)
        The unique classes in the training data.

    n_features_in_ : int
        The number of features in the training data.

    X_train_ : ndarray, shape (n_samples, n_features)
        The training data, stored for distance computation during prediction.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor model to the training data.

        Has the following parameters:
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input training data.

        y : ndarray of shape (n_samples,)
            The target labels for the training data.

        Returns:
        -------
        self : OneNearestNeighbor
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.X_train_ = X
        self.y_train_ = y

        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Has the following parameters:
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data for which to predict the labels.

        Returns:
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted labels for each sample in X.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0],
                         dtype=self.classes_.dtype)

        for i, test_point in enumerate(X):
            distances = np.linalg.norm(self.X_train_ - test_point, axis=1)
            nearest_neighbor_index = np.argmin(distances)
            y_pred[i] = self.y_train_[nearest_neighbor_index]
        return y_pred

    def score(self, X, y):
        """Compute the accuracy of the model on the given data.

        Has the following parameters:
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data for which to compute the accuracy.

        y : ndarray of shape (n_samples,)
            The true labels for the input data.

        Returns:
        -------
        accuracy : float
            The fraction of correctly classified samples.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        accuracy = np.mean(y_pred == y)
        return accuracy
