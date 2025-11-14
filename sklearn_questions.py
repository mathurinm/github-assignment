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
from sklearn.utils.validation import validate_data


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """
    One-nearest-neighbor classifier.
    This estimator predicts the label of each input sample as the label of
    the single closest training sample under the Euclidean distance.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """
        Fit the classifier.
        The fitting process for OneNearestNeighbor only means storing the training data,
        as it is a lazy learning algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.
        Returns
        -------
        self :
        OneNearestNeighbor Fitted estimator.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """
        Predict class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, attributes=("X_", "y_"))
        X = validate_data(
            self, X, reset=False, dtype=np.float64, ensure_all_finite=True)
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)
        Xt_sq = np.sum(self.X_ ** 2, axis=1, keepdims=True).T
        dist_2 = X_sq + Xt_sq - 2 * (X @ self.X_.T)
        nn_index = np.argmin(dist_2, axis=1)
        y_pred = self.y_[nn_index]

        return y_pred

    def score(self, X, y):
        """
        Return accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Test samples.
        y : array-like of shape (n_samples,)
        True labels.

        Returns
        -------
        float
        Accuracy of ``self.predict(X)`` vs ``y``.
        """
        check_is_fitted(self, attributes=("X_", "y_"))
        X = validate_data(
            self, X, reset=False, dtype=np.float64, ensure_all_finite=True)
        y = check_array(y, ensure_2d=False)

        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y have incompatible shapes.")

        y_pred = self.predict(X)

        return float(np.mean(y_pred == y))
