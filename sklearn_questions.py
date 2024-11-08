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

    Implementation of the 1-Nearest Neighbor algorithm. It assigns
    the class of the nearest training sample to each test sample based on
    the Euclidean distance.

    Attributes:
    -----------
    X_train_ : array-like of shape (n_samples, n_features)
        The training input samples.
    y_train_ : array-like of shape (n_samples,)
        The target values.
    classes_ : array-like of shape (n_classes,)
        The class labels.
    n_features_in_ : int
        The number of features seen during fit.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        --------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.X_train_ = X
        self.y_train_ = y

        # XXX fix
        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )
        for i, sample in enumerate(X):
            dist = np.linalg.norm(self.X_train_ - sample, axis=1)
            index = np.argmin(dist)
            y_pred[i] = self.y_train_[index]

        # XXX fix
        return y_pred

    def score(self, X, y):
        """Return the accuracy on the given test data and labels.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns:
        --------
        score : float
            Average number of samples corectly classified
            by self.predict(X) with respect to y.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        # XXX fix
        return np.mean(y == y_pred)
