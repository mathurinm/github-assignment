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
        """
        Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : array of shape (n_samples, n_features_in)
            The training input samples.
        y : array of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Returns the fitted instance.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """
        Predict class labels for the given input samples.

        Parameters
        ----------
        X : array of shape (n_samples, n_features_in)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self)  # Verify that the estimator is fitted
        X = check_array(X)
        y_pred = np.full(
            shape=len(X),
            fill_value=self.classes_[0],
            dtype=self.classes_.dtype,
        )
        for i, x in enumerate(X):
            distances = np.linalg.norm(self.X_train_ - x, axis=1)
            nearest_index = np.argmin(distances)
            y_pred[i] = self.y_train_[nearest_index]
        return y_pred

    def score(self, X, y):
        """
        Compute the accuracy of the classifier.

        Parameters
        ----------
        X : array of shape (n_samples, n_features_in)
            The input samples.

        y : array of shape (n_samples,)
            True labels for the input samples.

        Returns
        -------
        accuracy : float
            The fraction of correctly classified samples.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
