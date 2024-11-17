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
    "OneNearestNeighbor classifier."

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Write docstring.

        And describe parameters
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # XXX fix
        self._X_train = np.array(X)
        self._y_train = np.array(y)

        return self

    def predict(self, X):
        """
        Predict the class labels for the input samples.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            The input samples for which to predict labels.

        Returns:
        - y_pred: np.ndarray, shape (n_samples,)
            Predicted class labels for each input sample.
        """
        check_is_fitted(self)
        X = check_array(X) 
        n_samples = X.shape[0]
        n_train_samples = self._X_train.shape[0]

        # compute the distance between each element of X and X_train
        dist = np.linalg.norm(X[:, np.newaxis] - self._X_train, axis=2)

        # Find the nearest neighboor
        nearest_indices = np.argmin(dist, axis=1).astype(int)

        y_pred = self._y_train[nearest_indices]
        return y_pred



    def score(self, X, y):
        """
        Compute the accuracy of the classifier.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            The input samples for testing.
        - y: np.ndarray, shape (n_samples,)
            True labels for the input samples.

        Returns:
        - result: float
            The result of the predictions compared to the true labels.
        """
        X, y = check_X_y(X, y)

        y_pred = self.predict(X)
        result = np.mean(y_pred == y)
        return result
