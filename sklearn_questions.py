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
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Initialize the class object.

        Parameters
        ----------
        X : ndarry of shape (n_samples, n_features)
            The input array of training data.
        y : ndarry of shape (n_samples)
            Target labels.

        Returns
        -------
        self : The initialized object.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        # XXX fix
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Return the perdiction array.

        Parameters
        ----------
        self : The object.
        X : ndarray of shape (n_samples, n_features)
            The input array (training data).

        Returns
        -------
        y_pred : ndarray of shape (n_samples, n_features)
                 Predicted labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        # XXX fix
        distance = euclidean_distances(X, self.X_)
        closest_neighbor_indices = np.argmin(distance, axis=1)
        y_pred = self.y_[closest_neighbor_indices]

        return y_pred

    def score(self, X, y):
        """Compute prediction score.

        Parameters
        ----------
        self : The object.
        X : ndarray of shape (n_samples, n_features)
            The input array (training data).

        y : ndarray of shape (n_samples, n_features)
            The expected lables (targets).

        Returns
        -------
        score : float
            The 'score' of the accuracy of the prediction.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        # XXX fix
        score = accuracy_score(y_pred, y)

        return score
