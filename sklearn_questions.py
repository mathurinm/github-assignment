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
        Fit the OneNearestNeighbor classifier 

        And describe parameters:
        X : array-like of shape (samples, features)
            Training vectors, where samples is the number of samples
            and features is the number of features.
        y : array-like of shape (samples,)
            Target values (class labels in classification).

        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.X = X
        self.y = y

        return self

    def predict(self, X):
        """Write docstring.
        Predict the class labels

        And describe parameters:
        X : array-like of shape (observations, features)
            Test vectors, where observations is the number of samples
            and features is the number of features.
        y_pred : ndarray of shape (observations,)
            Predicted class labels for each data sample.

        """
        check_is_fitted(self)
        X = check_array(X)
        n_count = X.shape[0]
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )
        for i in range(n_count):
            distances = []
            for j in range(self.X.shape[0]):
                distance = np.sqrt(np.sum((X[i] - self.X[j]) ** 2))
                distances.append(distance)
            index = np.argmin(distances)
            y_pred[i] = self.y_[index]

        return y_pred




    def score(self, X, y):
        """Write docstring.
        Compute the accuracy of the model

        And describe parameters
        X : array-like of shape (samples, n_features)
            Test samples.
        y : array-like of shape (samples,)
            True labels for X.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        return np.mean(y_pred == y)
