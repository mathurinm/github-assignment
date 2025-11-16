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

    This estimator implements the 1-Nearest Neighbor classification algorithm.
    It predicts the label of a test sample based on the label of the single
    closest training sample (using Euclidean distance).

    No hyperparameters are needed for 1-NN.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        The 1-NN model simply stores the training data (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : OneNearestNeighbor
            The fitted estimator.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.X_fit_ = X
        self.y_fit_ = y
        return self

    def predict(self, X):
        """Predict the class labels for the input samples.

        For each test sample, find the closest training sample using
        Euclidean distance and return its label.

        Parameters
        ----------
        X : array-like of shape (n_samples_test, n_features)
            The input samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples_test,)
            The predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)

        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        n_test = X.shape[0]

        for i in range(n_test):
            x_test = X[i, :]

            distances = np.sum((self.X_fit_ - x_test) ** 2, axis=1)

            nearest_neighbor_index = np.argmin(distances)

            y_pred[i] = self.y_fit_[nearest_neighbor_index]
        return y_pred

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples_test, n_features)
            The input samples.
        y : array-like of shape (n_samples_test,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        is_correct = (y_pred == y)

        n_samples = len(y)
        if n_samples == 0:
            y_pred = np.array([0.0])
            return y_pred.sum()

        y_pred = is_correct.astype(float) / n_samples

        return y_pred.sum()
