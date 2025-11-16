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
        """Fit the OneNearestNeighbor classifier on training data.

        Parameters
        ----------
        self : defines the instance of the class OneNearestNeighbor we are
        working on
        X : ndarray of the training data,
        with shape (n_observations, p_features)
        y : 1-darray of the labels associated with each dimension of X,
        with shape (n_observations)

        Returns
        -------
        self : maintains the instance
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # XXX fix
        self.X_ = X
        self.y_ = y #  store X and y as "learned" data,
        #  ensure we have trained on X and y
        return self

    def predict(self, X):
        """Predict y label for input data X using the OneNearestNeighbor rule.

        Parameters
        -------
        self : still maintain the instance of the class
        X : ndarray of test data, with shape (n_observations, p_features)

        Returns
        -------
        y_pred : 1-darray of labels predicted for the test data X,
        with shape (n_observations,)
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        # XXX fix
        idx = 0
        for x in X:
            euclidean_distances = np.sqrt(np.sum((self.X_ - x) ** 2, axis=1))
            NN_index = np.argmin(euclidean_distances)
            NN = self.y_[NN_index]
            y_pred[idx] = NN
            idx += 1
        return y_pred

    def score(self, X, y):
        """Score model performance by evaluating proportion of accurate y_pred.

        Parameters
        -------
        X : ndarray of test data, with shape (n_observations, p_features)
        y : 1d-array of the true labels associated with test samples X,
        with shape (n_observations,)

        Returns
        -------
        a score : float in [0,1]
        reflecting the proportion of accurate predictions
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        # XXX fix
        return sum(y_pred == y)/len(y)
