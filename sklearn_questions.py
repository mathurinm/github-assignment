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
from scipy.spatial.distance import euclidean


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier.

    This classifier implements the 1-NN algorithm.
    The target class of a given sample is the class of the closest training
    sample in feature space, measured by Euclidean distance.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the 1NN classifier according to the given training data.

        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # XXX fix
        return self

    def predict(self, X):
        """Perform classification on an array of test vectors X.

        X : array-like, shape (n_samples, n_features)
            Test samples.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.array([self.closest_sample(x) for x in X])

        # XXX fix
        return y_pred

    def closest_sample(self, x):
        """Find the class label of the closest training sample to x.

        x : array-like, shape (n_features,)
            A single input sample.
        """
        distances = [euclidean(x, train_x) for train_x in self.X_]
        min_index = np.argmin(distances)
        return self.y_[min_index]

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        X : array-like, shape (n_samples, n_features)
            Test samples.
        y : array-like, shape (n_samples,)
            True labels for X.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        # XXX fix
        return y_pred.sum()
