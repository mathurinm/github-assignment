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
from scipy.spatial.distance import cdist


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fits the classifier with thetraining data X and labels y.

        Parameters
        ----------
        X (ndarray) : Shape (n, d) ; the coordinates of the training points
        y (ndarray) : Shape (n, 1) ; the labels of the training points

        Returns
        -------
        self (OneNearestNeighbor) : The fitted classifier
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predicts the lables of the points in X.

        Parameters
        ----------
        X (ndarray) : Shape (n, d) ; Coordinates of points to
                                     predict the label of.

        Returns
        -------
        y_pred (ndarray) : Shape (n, 1) ; Predicted labels of the points in X
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        distances = cdist(X, self.X_)
        index_min_distancs_array = distances.argmin(axis=1)
        y_pred = self.y_[index_min_distancs_array]
        return y_pred

    def score(self, X, y):
        """Compute accuracy score of the classifier on the provided test base.

        Parameters
        ----------
        X (ndarray) : Shape (n, d) ; the coordinates of the test points
        y (ndarray) : Shape (n, 1) ; the corresponding
                                     lables of the test points

        Returns
        -------
        accuracy (float) : The accuracy score on the provided test base
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        y_pred = y_pred == y
        return y_pred.sum() / len(y_pred)
