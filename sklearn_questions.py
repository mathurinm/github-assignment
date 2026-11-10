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
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import validate_data


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """One-nearest-neighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the classifier to the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray of shape (n_samples,)
            Training class labels.

        Returns
        -------
        self : OneNearestNeighbor
            Fitted classifier.
        """
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.X_train_ = X
        self.y_train_ = y

        return self

    def predict(self, X):
        """Predict class labels using the nearest neighbor.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Observations to classify.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        y_pred = np.full(
            shape=len(X),
            fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        for i, x in enumerate(X):
            distances = np.linalg.norm(self.X_train_ - x, axis=1)
            index = np.argmin(distances)
            y_pred[i] = self.y_train_[index]

        return y_pred

    def score(self, X, y):
        """Compute the classification accuracy.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Observations to classify.
        y : ndarray of shape (n_samples,)
            True class labels.

        Returns
        -------
        score : float
            Proportion of correctly classified observations.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        y_pred = (y_pred == y) / len(y)

        return y_pred.sum()
