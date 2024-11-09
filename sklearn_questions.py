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
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit a 1-nearest neighbor classifier (memorize training data).

        Parameters
        ----------
        X : ndarray
            A numpy array of shape (num_train, p) containing training data
            consisting of num_train samples each with dimension p

        y : ndarray
            A numpy array of shape (num_train, ) containing training labels

        Returns
        -------
        self: the classifier itself
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # XXX fix
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predict labels for test data using this classifier.

        Parameters
        ----------
        X : ndarray
            A numpy array of shape (num_test, p) containing test data
            each with dimension p

        Returns
        -------
        y_pred : ndarray
            A numpy array of shape (num_test, ) containing predicted labels
            of test data
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        for i, x in enumerate(X):
            dis = np.linalg.norm(x - self.X_train_, axis=1)
            closest_y_index = np.argmin(dis)
            y_pred[i] = self.y_train_[closest_y_index]

        # XXX fix
        return y_pred

    def score(self, X, y):
        """Calculate the accuracy of the label predictions.

        Parameters
        ----------
        X : ndarray
            A numpy array of shape (num_test, p) containing test data
            each with dimension p

        y : ndarray
            A numpy array of shape (num_test, ) containing num_test
            test labels

        Returns
        -------
        accuracy : float64
            The accuracy of the predictions.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        # XXX fix
        accuracy = np.mean(y_pred == y)
        return accuracy
