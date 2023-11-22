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
        """Train a OneNearestNeighbor classifier. This is just memorizing the
        data.
        Parameters
        ----------
        X : ndarray of shape (num_train, D).
            The input array containing the training data consisting of
        num_train samples with flatten size D.

        y : ndarray of shape (num_train,)
            The trainnig labels associate to the input array, where y[i] is
            the label for X[i].
        Returns
        -------
        X_train : ndarray of shape (num_train, D)
            Copy of X
        y_train : ndarray of shape (num_train,)
            Copy of y
        """

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.n_features_in_ = X.shape[1]
        # XXX fix
        return self

    def predict(self, X):
        """Predict labels for test data using the fitted classifier. Gives
    label of the closest point using Euclidean distance.
        Parameters
        ----------
        X : ndarray of shape (num_test, D)
            The input array containing the test data consisting of num_test
            samples with flatten size D.
        Returns
        -------
        y : ndarray of shape (num_test,)
            Predicted labels associated to the test data, where y[i] is the
            predicted label for X[i].
        """

        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )
        A = np.diag(X @ X.T)[:, np.newaxis]
        B = np.diag(self.X_train_ @ self.X_train_.T)[np.newaxis, :]
        C = X @ self.X_train_.T
        dists = np.sqrt(A + B - 2 * C)
        nearest = np.argsort(dists)[:, 0]
        for i in range(len(X)):
            y_pred[i] = self.y_train_[nearest[i]]
        # XXX fix
        return y_pred

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        And describe parameters

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        accuracy : float
            Mean accuracy of self.predict(X) w.r.t. y.
        """
        X, y = check_X_y(X, y)
        accuracy = np.mean(self.predict(X) == y)
        # XXX fix
        return accuracy
