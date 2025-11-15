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
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import validate_data


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the model based on X and y.

        Parameters
        ----------
        X: array (n_samples, n_features). Training data.

        y: array (n_samples). Target data.

        Returns
        -------
        self: object

        """
        X, y = validate_data(self, X, y, dtype=float)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict the labels for each x based on Euclidean distance.

        Parameters
        ----------
        X: array of test samples

        Returns
        -------
        y_pred: array with shape (n_samples, ) of predicted labels
        """
        check_is_fitted(self)
        X = validate_data(self, X, dtype=float, reset=False)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        # Compute distances
        for i, x_test in enumerate(X):
            distances = np.linalg.norm(self.X_ - x_test, axis=1)
            idx = np.argmin(distances)
            y_pred[i] = self.y_[idx]

        return y_pred

    def score(self, X, y):
        """Return the accuracy of the classifier.

        Parameters
        ----------
        X: array (n_samples, n_features). Training data.

        y: array (n_samples). Target data.

        y_pred: array with shape (n_samples, ) of predicted labels

        Returns
        -------
        accuracy: float, fraction of correct predictions.
        """
        X, y = validate_data(self, X, y, dtype=float, reset=False)
        y_pred = self.predict(X)

        return np.mean(y_pred == y)
