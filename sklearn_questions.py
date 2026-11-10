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
    "OneNearestNeighbor classifier."

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Stores the training data to fit the classifier.

        Parameters
        ----------
        X = ndarray of shape (n_samples, n_features)
        The training sample, input array.

        y = ndarray of shape (n_samples,)
        The training target, input array.

        Returns
        -------
        self : OneNearestNeighbor
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predicts the target of each sample.

        Parameters
        ----------
        X = ndarray of shape (n_samples, n_features)
        The training sample, input array.

        Returns
        -------
        y_pred : nd array of shape (n_samples, )
        The target of the closest training sample for each sample.

        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        # XXX fix

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input"
            )
        for i, x in enumerate(X):
            distances = np.sum((self.X_train_ - x) ** 2, axis=1)
            y_pred[i] = self.y_train_[np.argmin(distances)]

        return y_pred

    def score(self, X, y):
        """Computes the accuracy of the model.

        Parameters
        ----------
        X = ndarray of shape (n_samples, n_features)
        The training sample, input array.

        y = ndarray of shape (n_samples,)
        The training target, input array.

        Returns
        -------
        score : float
        The fraction of the samples correctly classified
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        # XXX fix
        return np.mean(y_pred == y)
