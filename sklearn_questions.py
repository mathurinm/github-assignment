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


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        This method stores the training data so that predictions can be
        made by finding, for each test sample, the closest training sample
        in Euclidean distance.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data. Each row corresponds to one sample and each
            column corresponds to one feature.

        y : ndarray of shape (n_samples,)
        Target labels for the training samples. Must be a classification
        target (e.g. integers or strings).

        Returns
        -------
        self : OneNearestNeighbor
            The fitted classifier.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : ndarray of shape (n_test_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : ndarray of shape (n_test_samples,)
            Predicted labels.
        """
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input"
            )

        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        euc_dist = np.sqrt(((X[None, :, :] - self.X_[:, None, :])**2)
                           .sum(axis=2))
        nearest_point = np.argmin(euc_dist, axis=0)
        y_pred = self.y_[nearest_point]

        return y_pred

    def score(self, X, y):
        """Return accuracy on test samples after running predict().

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.

        y : ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        accuracy : float
            Fraction of correctly classified samples.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        return np.mean(y_pred == y)
